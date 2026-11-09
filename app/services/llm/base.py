from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum


class ModelAccess(StrEnum):
    FREE = "free"  # usable on the current plan
    PAID = "paid"  # needs credits / paid plan
    UNUSABLE = "unusable"  # not a chat model, or permanently failing
    UNKNOWN = "unknown"  # transient error, retry later


@dataclass
class LLMMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResult:
    text: str
    model: str
    usage: LLMUsage


@dataclass
class LLMChunk:
    """Streaming event: either a text delta or the final usage (done=True)."""

    delta: str = ""
    done: bool = False
    usage: LLMUsage | None = None


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None,
        model: str | None = None,
    ) -> LLMResult: ...

    @abstractmethod
    def stream(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None,
        model: str | None = None,
    ) -> AsyncIterator[LLMChunk]: ...

    @abstractmethod
    async def list_models(self) -> list[str]: ...

    @abstractmethod
    async def check_model(self, model: str) -> ModelAccess: ...
