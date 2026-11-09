"""Query rewriting, relevance threshold and history trimming in ChatService."""

import asyncio
import json
import logging

import pytest

from app.core.config import settings
from app.services.chat_service import (
    NO_RELEVANT_SOURCES_NOTE,
    RAG_INSTRUCTIONS,
    ChatService,
    trim_history,
)
from app.services.chunking import count_tokens
from app.services.llm.base import LLMMessage, LLMResult, LLMUsage
from app.services.query_rewriter import REWRITE_SYSTEM_PROMPT
from tests.fakes import DOC_ID, FakeProvider, FakeRetriever, make_row


class RewritingProvider(FakeProvider):
    """Answers rewrite requests with a scripted query (or failure), everything else
    like FakeProvider."""

    def __init__(self, rewrite: str | Exception = "сколько спят кошки", delay=0.0):
        super().__init__()
        self.rewrite = rewrite
        self.delay = delay
        self.rewrite_calls: list[dict] = []

    async def generate(self, messages, system_prompt, model=None, temperature=None):
        if system_prompt != REWRITE_SYSTEM_PROMPT:
            return await super().generate(messages, system_prompt, model)
        self.rewrite_calls.append(
            {"prompt": messages[0].content, "model": model, "temperature": temperature}
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(self.rewrite, Exception):
            raise self.rewrite
        return LLMResult(self.rewrite, model or "m", LLMUsage(3, 2))


async def _follow_up(
    svc, chat, first="Расскажи про кошек", second="А сколько они спят?"
):
    await svc.send(chat, first)
    return await svc.send(chat, second)


# ---- 1. query rewriting -----------------------------------------------------------


async def test_follow_up_is_searched_by_the_rewritten_query(db, chat):
    db.document_ids = [DOC_ID]
    provider = RewritingProvider("сколько спят кошки")
    retriever = FakeRetriever([make_row()])

    await _follow_up(ChatService(None, provider, retriever), chat)

    # the search used the standalone query...
    assert retriever.calls[-1][1] == "сколько спят кошки"
    # ...but the model was asked the user's original question
    answer_call = provider.calls[-1]
    assert answer_call["messages"][-1].content == "А сколько они спят?"
    # rewrite request: deterministic, configured model, sees the previous turn
    call = provider.rewrite_calls[0]
    assert call["temperature"] == 0
    assert call["model"] == (settings.rewrite_model or settings.llm_model)
    assert "Расскажи про кошек" in call["prompt"]
    assert "А сколько они спят?" in call["prompt"]


async def test_first_question_is_not_rewritten(db, chat):
    db.document_ids = [DOC_ID]
    provider = RewritingProvider()
    retriever = FakeRetriever([make_row()])

    await ChatService(None, provider, retriever).send(chat, "Сколько спят кошки?")

    assert provider.rewrite_calls == []
    assert retriever.calls[0][1] == "Сколько спят кошки?"


async def test_chat_without_documents_is_not_rewritten(db, chat):
    provider = RewritingProvider()

    await _follow_up(ChatService(None, provider, FakeRetriever([])), chat)

    assert provider.rewrite_calls == []


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("LLM is down"), "", "   \n  "],
    ids=["exception", "empty", "blank"],
)
async def test_rewrite_failure_falls_back_to_the_original_question(
    db, chat, caplog, failure
):
    db.document_ids = [DOC_ID]
    retriever = FakeRetriever([make_row()])
    svc = ChatService(None, RewritingProvider(failure), retriever)

    with caplog.at_level(logging.WARNING, logger="app.services.query_rewriter"):
        reply = await _follow_up(svc, chat)

    assert retriever.calls[-1][1] == "А сколько они спят?"
    assert reply.content  # the answer still went through
    assert any(r.levelno == logging.WARNING for r in caplog.records)


async def test_rewrite_timeout_falls_back(db, chat, monkeypatch):
    monkeypatch.setattr(settings, "rewrite_timeout_s", 0.01)
    db.document_ids = [DOC_ID]
    retriever = FakeRetriever([make_row()])
    svc = ChatService(None, RewritingProvider("slow", delay=1), retriever)

    await _follow_up(svc, chat)

    assert retriever.calls[-1][1] == "А сколько они спят?"


async def test_rewrite_is_logged(db, chat, caplog):
    db.document_ids = [DOC_ID]
    svc = ChatService(None, RewritingProvider("сон кошек"), FakeRetriever([make_row()]))

    with caplog.at_level(logging.INFO, logger="app.services.query_rewriter"):
        await _follow_up(svc, chat)

    assert any(
        "А сколько они спят?" in r.getMessage() and "сон кошек" in r.getMessage()
        for r in caplog.records
    )


async def test_rewritten_query_is_cleaned(db, chat):
    db.document_ids = [DOC_ID]
    retriever = FakeRetriever([make_row()])
    provider = RewritingProvider('"Поисковый запрос: сон кошек"\nПояснение: ...')

    await _follow_up(ChatService(None, provider, retriever), chat)

    assert retriever.calls[-1][1] == "сон кошек"


async def test_rewriter_sees_only_the_last_6_messages(db, chat):
    db.document_ids = [DOC_ID]
    provider = RewritingProvider()
    svc = ChatService(None, provider, FakeRetriever([make_row()]))

    for i in range(5):  # 5 questions + 5 answers before the follow-up
        await svc.send(chat, f"вопрос номер {i}")
    provider.rewrite_calls.clear()
    await svc.send(chat, "и что дальше?")

    prompt = provider.rewrite_calls[0]["prompt"]
    assert "вопрос номер 0" not in prompt and "вопрос номер 1" not in prompt
    assert "вопрос номер 2" in prompt and "вопрос номер 4" in prompt


# ---- 2. relevance threshold -------------------------------------------------------


async def test_chunks_below_threshold_are_dropped(db, chat, monkeypatch):
    monkeypatch.setattr(settings, "rag_min_score", 0.4)
    db.document_ids = [DOC_ID]
    provider = FakeProvider()
    rows = [
        make_row("Cats sleep 16 hours.", page=1, score=0.9),
        make_row("Unrelated tax law.", page=2, score=0.35),
        make_row("Kittens sleep more.", page=3, score=0.6),
    ]

    reply = await ChatService(None, provider, FakeRetriever(rows)).send(chat, "q")

    system = provider.calls[0]["system"]
    assert "Unrelated tax law." not in system
    assert "[1] cats.pdf, стр. 1" in system and "[2] cats.pdf, стр. 3" in system
    assert [(s["n"], s["page"]) for s in reply.sources] == [(1, 1), (2, 3)]


async def test_score_exactly_at_threshold_is_kept(db, chat, monkeypatch):
    monkeypatch.setattr(settings, "rag_min_score", 0.4)
    db.document_ids = [DOC_ID]

    reply = await ChatService(
        None, FakeProvider(), FakeRetriever([make_row(score=0.4)])
    ).send(chat, "q")

    assert len(reply.sources) == 1


async def test_nothing_relevant_means_no_sources_block(db, chat, monkeypatch):
    monkeypatch.setattr(settings, "rag_min_score", 0.4)
    db.document_ids = [DOC_ID]
    provider = FakeProvider()
    rows = [make_row(score=0.2), make_row(score=0.39)]

    reply = await ChatService(None, provider, FakeRetriever(rows)).send(chat, "q")

    system = provider.calls[0]["system"]
    assert "<sources>" not in system
    assert RAG_INSTRUCTIONS not in system
    assert NO_RELEVANT_SOURCES_NOTE in system
    assert system.startswith("You are a cat expert.")
    assert reply.sources == []  # documents exist, so [] and not None


async def test_stream_sends_empty_sources_event_when_nothing_relevant(db, chat):
    db.document_ids = [DOC_ID]
    svc = ChatService(None, FakeProvider(), FakeRetriever([make_row(score=0.1)]))

    events = [e async for e in svc.stream(chat, "q")]

    assert events[0] == ("sources", "[]")
    assert json.loads(events[-1][1])["sources"] == []


# ---- 3. history trimming ----------------------------------------------------------


def _msg(role: str, words: int, tag: str = "") -> LLMMessage:
    return LLMMessage(role=role, content=f"{tag} " + "слово " * words)


def _dialog(turns: int, words: int = 50) -> list[LLMMessage]:
    out = []
    for i in range(turns):
        out.append(_msg("user", words, f"q{i}"))
        out.append(_msg("assistant", words, f"a{i}"))
    out.append(_msg("user", 10, "last"))
    return out


def _tokens(messages: list[LLMMessage]) -> int:
    return sum(count_tokens(m.content) for m in messages)


def test_trim_history_respects_the_budget():
    messages = _dialog(20)
    budget = 600

    kept = trim_history(messages, budget)

    assert _tokens(kept) <= budget
    assert len(kept) < len(messages)
    # what is kept is the most recent, contiguous tail, in order
    assert kept == messages[len(messages) - len(kept) :]


def test_trim_history_keeps_everything_that_fits():
    messages = _dialog(2, words=5)
    assert trim_history(messages, 10_000) == messages


def test_trim_history_always_keeps_the_last_message():
    huge = _msg("user", 5000, "huge question")
    messages = [*_dialog(3), huge]

    kept = trim_history(messages, 100)

    assert kept == [huge]  # over budget on its own, but never dropped


def test_trim_history_never_starts_with_assistant():
    messages = _dialog(10)
    # find budgets where a naive cut would start on an assistant reply
    for budget in range(50, 1500, 7):
        kept = trim_history(messages, budget)
        assert kept[0].role == "user", budget
        assert kept[-1] is messages[-1]


def test_trim_history_empty():
    assert trim_history([], 100) == []


async def test_chat_service_sends_trimmed_history(db, chat, monkeypatch):
    provider = FakeProvider()
    svc = ChatService(None, provider, FakeRetriever([]))
    for i in range(6):
        await svc.send(chat, f"вопрос {i} " + "длинный текст " * 40)
    monkeypatch.setattr(settings, "history_max_tokens", 300)

    await svc.send(chat, "последний вопрос")

    sent = provider.calls[-1]["messages"]
    assert sent[-1].content == "последний вопрос"
    assert sent[0].role == "user"
    assert _tokens(sent) <= 300
    assert len(sent) < len(db.messages)
