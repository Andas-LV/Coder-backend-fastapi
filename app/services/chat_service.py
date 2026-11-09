import time
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Chat, Message
from app.repositories import chat as repo
from app.schemas.chat import MessageOut
from app.services.llm.base import LLMMessage, LLMProvider, LLMUsage
from app.services.llm.pricing import calc_cost


class ChatService:
    def __init__(self, session: AsyncSession, provider: LLMProvider) -> None:
        self.session = session
        self.provider = provider

    async def _prepare(
        self, chat: Chat, content: str
    ) -> tuple[list[LLMMessage], str, str]:
        """Save the user message; return (history, system prompt, model)."""
        # Capture plain values first: commits expire ORM attributes.
        chat_id = chat.id
        system_prompt = chat.system_prompt or settings.default_system_prompt
        model = chat.model or settings.llm_model
        await repo.add_message(self.session, chat_id, "user", content)
        history = await repo.get_history(self.session, chat_id)
        messages = [
            LLMMessage(role=m.role, content=m.content)
            for m in history
            if m.role != "system"
        ]
        return messages, system_prompt, model

    async def _save_reply(
        self, chat_id, text: str, model: str, usage: LLMUsage, started: float
    ) -> Message:
        return await repo.add_message(
            self.session,
            chat_id,
            "assistant",
            text,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
            cost_usd=calc_cost(model, usage.input_tokens, usage.output_tokens),
        )

    async def send(self, chat: Chat, content: str) -> Message:
        chat_id = chat.id
        messages, system_prompt, model = await self._prepare(chat, content)
        started = time.perf_counter()
        result = await self.provider.generate(messages, system_prompt, model)
        return await self._save_reply(
            chat_id, result.text, model, result.usage, started
        )

    async def stream(self, chat: Chat, content: str) -> AsyncIterator[tuple[str, str]]:
        """Yield SSE (event, data) pairs: 'delta' per chunk, then 'done' with the saved message."""
        chat_id = chat.id
        messages, system_prompt, model = await self._prepare(chat, content)
        started = time.perf_counter()
        parts: list[str] = []
        usage = LLMUsage()
        async for chunk in self.provider.stream(messages, system_prompt, model):
            if chunk.delta:
                parts.append(chunk.delta)
                yield "delta", chunk.delta
            if chunk.done and chunk.usage:
                usage = chunk.usage
        msg = await self._save_reply(chat_id, "".join(parts), model, usage, started)
        yield "done", MessageOut.model_validate(msg).model_dump_json()
