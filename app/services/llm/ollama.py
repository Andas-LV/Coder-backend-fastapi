from collections.abc import AsyncIterator

from ollama import AsyncClient, ChatResponse, ResponseError

from app.core.config import settings
from app.services.llm.base import (
    LLMChunk,
    LLMMessage,
    LLMProvider,
    LLMResult,
    LLMUsage,
    ModelAccess,
)


def _messages(
    messages: list[LLMMessage], system_prompt: str | None
) -> list[dict[str, str]]:
    out = [{"role": "system", "content": system_prompt}] if system_prompt else []
    out += [{"role": m.role, "content": m.content} for m in messages]
    return out


def _usage(resp: ChatResponse) -> LLMUsage:
    return LLMUsage(
        input_tokens=resp.prompt_eval_count or 0,
        output_tokens=resp.eval_count or 0,
    )


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        self._client = AsyncClient(
            host=settings.ollama_host,
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"},
        )

    async def generate(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        model = model or settings.llm_model
        options = {"temperature": temperature} if temperature is not None else None
        resp = await self._client.chat(
            model=model, messages=_messages(messages, system_prompt), options=options
        )
        return LLMResult(
            text=resp.message.content or "", model=model, usage=_usage(resp)
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None,
        model: str | None = None,
    ) -> AsyncIterator[LLMChunk]:
        model = model or settings.llm_model
        usage = LLMUsage()
        async for part in await self._client.chat(
            model=model, messages=_messages(messages, system_prompt), stream=True
        ):
            if part.message.content:
                yield LLMChunk(delta=part.message.content)
            if part.done:
                usage = _usage(part)
        yield LLMChunk(done=True, usage=usage)

    async def list_models(self) -> list[str]:
        resp = await self._client.list()
        return sorted(m.model for m in resp.models if m.model)

    async def check_model(self, model: str) -> ModelAccess:
        """Probe with a 1-token request: Ollama has no 'is this model free' endpoint."""
        try:
            await self._client.chat(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1},
            )
        except ResponseError as e:
            msg = str(e).lower()
            if "free usage" in msg or e.status_code in (402, 403):
                return ModelAccess.PAID
            if e.status_code == 429 or e.status_code >= 500:
                return ModelAccess.UNKNOWN
            return ModelAccess.UNUSABLE
        except Exception:  # noqa: BLE001 - network errors etc. are transient
            return ModelAccess.UNKNOWN
        return ModelAccess.FREE
