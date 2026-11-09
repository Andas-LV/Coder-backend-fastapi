"""Turn a follow-up question into a standalone search query.

"А что во втором разделе?" means nothing to a vector index on its own; with the last few
messages as context the LLM can rewrite it into "второй раздел договора: сроки оплаты".
The rewritten query is used only for retrieval; the model still answers the original
question.
"""

import asyncio
import logging

from app.core.config import settings
from app.services.llm.base import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

HISTORY_MESSAGES = 6  # how much of the conversation the rewriter sees
_MAX_MESSAGE_CHARS = 1500  # long answers are cut: the gist is enough for rewriting
_MAX_QUERY_CHARS = 500

REWRITE_SYSTEM_PROMPT = """\
Ты переписываешь последний вопрос пользователя в самостоятельный поисковый запрос \
для поиска по документам.
- Используй историю диалога, чтобы раскрыть местоимения и отсылки ("он", "второй пункт", \
"а что там про сроки").
- Пиши на языке вопроса.
- Не отвечай на вопрос и ничего не объясняй.
- Верни только сам запрос, одной строкой, без кавычек и префиксов."""


def _format_history(history: list[LLMMessage]) -> str:
    lines = []
    for m in history[-HISTORY_MESSAGES:]:
        who = "Пользователь" if m.role == "user" else "Ассистент"
        text = m.content.strip().replace("\n", " ")
        if len(text) > _MAX_MESSAGE_CHARS:
            text = text[:_MAX_MESSAGE_CHARS] + "…"
        lines.append(f"{who}: {text}")
    return "\n".join(lines)


def _clean(text: str) -> str:
    """First non-empty line, without quotes/labels the model may add anyway."""
    quotes = "\"'«»` "
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    line = line.strip(quotes)
    for prefix in ("Поисковый запрос:", "Запрос:", "Search query:", "Query:"):
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix) :]
            break
    return line.strip(quotes)[:_MAX_QUERY_CHARS]


async def rewrite_query(
    provider: LLMProvider, history: list[LLMMessage], question: str
) -> str:
    """Standalone search query for `question` given the previous `history`.

    `history` must not include the question itself. Never raises: on error, timeout or
    an empty answer it falls back to the original question.
    """
    prompt = (
        f"История диалога:\n{_format_history(history)}\n\n"
        f"Последний вопрос пользователя:\n{question}\n\n"
        "Поисковый запрос:"
    )
    try:
        result = await asyncio.wait_for(
            provider.generate(
                [LLMMessage(role="user", content=prompt)],
                REWRITE_SYSTEM_PROMPT,
                settings.rewrite_model or settings.llm_model,
                temperature=0,
            ),
            timeout=settings.rewrite_timeout_s,
        )
    except TimeoutError:
        logger.warning(
            "Query rewrite timed out after %ss, using the original question",
            settings.rewrite_timeout_s,
        )
        return question
    except Exception:
        logger.warning(
            "Query rewrite failed, using the original question", exc_info=True
        )
        return question

    query = _clean(result.text)
    if not query:
        logger.warning("Query rewrite returned nothing, using the original question")
        return question
    logger.info("Query rewrite: %r -> %r", question, query)
    return query
