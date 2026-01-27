import asyncio
import logging
import os
from typing import Any
from typing import cast
from typing import Literal
from typing import TYPE_CHECKING

from jfbench.imports import LazyImport


try:
    import vllm
except ImportError:
    vllm = LazyImport("vllm")

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion import Choice as ChatChoice
from openai.types.completion import Completion as LegacyCompletion
from openai.types.completion import CompletionChoice as LegacyChoice
from tqdm import tqdm


if TYPE_CHECKING:
    from collections.abc import Sequence

N_PARALLEL_REQUEST = 200
N_RETRIES_PER_REQUEST = 3

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        provider: Literal["openrouter", "local", "vllm"] = "openrouter",
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        body_params = dict(extra_body) if extra_body is not None else {}
        self.client: AsyncOpenAI | vllm.LLM
        self.semaphore: asyncio.Semaphore | None = None
        model_name = model or "openai/gpt-oss-120b"
        if provider == "local":
            tensor_parallel_size = body_params.pop("tensor_parallel_size", 1)
            self.client = vllm.LLM(
                model=model_name,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
            )
        elif provider == "openrouter":
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
            self.semaphore = asyncio.Semaphore(N_PARALLEL_REQUEST)
        elif provider == "vllm":
            base_url = body_params.pop("base_url", "http://localhost:8000/v1")
            api_key = body_params.pop("api_key", "unsed")
            timeout = body_params.pop("timeout", 600)
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
            self.semaphore = asyncio.Semaphore(N_PARALLEL_REQUEST)
            print(f"Using vLLM endpoint at {base_url} for model {model_name}")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        self.provider = provider
        self.model = model_name
        self.temperature = body_params.pop("temperature", 0.0)
        self.max_tokens = body_params.pop("max_tokens", 4096)
        self.stop_token_ids = body_params.pop("stop_token_ids", None)
        self.extra_body = body_params

    async def async_ask(
        self,
        prompts: list[str],
        *,
        use_tqdm: bool = False,
    ) -> tuple[list[str], list[Any]]:
        if self.provider == "openrouter" or self.provider == "vllm":
            return await self._ask_openai(prompts, use_tqdm=use_tqdm)
        if self.provider == "local":
            return await asyncio.to_thread(self._ask_local, prompts, use_tqdm=use_tqdm)
        else:
            raise ValueError(f"Unsupported LLM provider for async_ask: {self.provider}")

    def ask(
        self,
        prompts: list[str],
        *,
        use_tqdm: bool = False,
    ) -> tuple[list[str], list[Any]]:
        if self.provider == "local":
            return self._ask_local(prompts, use_tqdm=use_tqdm)
        if self.provider == "openrouter" or self.provider == "vllm":
            return asyncio.run(self._ask_openai(prompts, use_tqdm=use_tqdm))
        else:
            raise ValueError(f"Unsupported LLM provider for ask: {self.provider}")

    def _ask_local(self, prompts: list[str], use_tqdm: bool) -> tuple[list[str], list[Any]]:
        assert self.stop_token_ids is not None
        sampling = vllm.SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop_token_ids=self.stop_token_ids,
            **self.extra_body,
        )
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        responses = list(
            self.client.chat(
                messages=messages,
                sampling_params=sampling,
                use_tqdm=use_tqdm,
            )
        )
        results: list[str] = []
        for i, r in enumerate(responses):
            results.append(r.outputs[0].text)
            if not r.finished:
                logger.warning(f"Generation for prompt {i} was not finished: {r.prompt}")
        return results, responses

    async def _ask_openai(
        self,
        prompts: list[str],
        *,
        use_tqdm: bool = False,
    ) -> tuple[list[str], list[Any]]:
        assert isinstance(self.client, AsyncOpenAI)
        assert self.semaphore is not None
        semaphore = self.semaphore

        async def _get_answer(index: int, prompt: str) -> tuple[int, str, Any]:
            messages = [{"role": "user", "content": prompt}]
            for i in range(N_RETRIES_PER_REQUEST):
                try:
                    async with semaphore:
                        response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            n=1,
                            extra_body=self.extra_body,
                        )
                    return index, to_string(response)[0].strip(), response
                except Exception as e:
                    logger.warning(
                        f"Error querying {self.provider} for prompt {index} (attempt {i + 1}): {e}"
                    )
                    if i == N_RETRIES_PER_REQUEST - 1:
                        raise e
            raise RuntimeError(
                f"Failed to obtain response for prompt {index} after {N_RETRIES_PER_REQUEST} retries."
            )

        tasks = [_get_answer(index, prompt) for index, prompt in enumerate(prompts)]
        results: list[str] = [""] * len(prompts)
        response_details: list[Any] = [None] * len(prompts)
        if tasks:
            iterator = asyncio.as_completed(tasks)
            progress = None
            if use_tqdm:
                progress = tqdm(
                    iterator,
                    total=len(tasks),
                    desc=f"Querying {self.provider}",
                    leave=False,
                )
                iterator = progress
            try:
                for coro in iterator:
                    idx, answer, response = await coro
                    results[idx] = answer
                    response_details[idx] = response
            finally:
                if progress is not None:
                    progress.close()
        return results, response_details


def to_string(obj: ChatCompletion | LegacyCompletion) -> list[str]:
    if isinstance(obj, ChatCompletion):
        raw_choices = getattr(obj, "choices", None)
        if not raw_choices:
            return [""]
        choices = cast("Sequence[ChatChoice]", raw_choices)
        return [(choice.message.content or "") for choice in choices]
    if isinstance(obj, LegacyCompletion):
        raw_choices = getattr(obj, "choices", None)
        if not raw_choices:
            return [""]
        choices = cast("Sequence[LegacyChoice]", raw_choices)
        return [(choice.text or "") for choice in choices]
    raise TypeError(f"Unsupported type: {type(obj)}")


def extract_reasoning_content(provider: str, response_detail: Any) -> str:
    if response_detail is None:
        return ""

    def _get_attr(obj: Any, name: str) -> Any:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict):
            return obj.get(name)
        return None

    try:
        choices = _get_attr(response_detail, "choices") or []
        if not choices:
            return ""
        first_choice = choices[0]
        message = _get_attr(first_choice, "message")
        if message is None:
            return ""
        if provider == "vllm":
            reasoning_content = _get_attr(message, "reasoning_content")
            if reasoning_content is not None:
                return str(reasoning_content)
            reasoning_content = _get_attr(_get_attr(message, "model_extra"), "reasoning_content")
            if reasoning_content is not None:
                return str(reasoning_content)
            return ""
        if provider == "openrouter":
            reasoning_content = _get_attr(message, "reasoning")
            if reasoning_content is not None:
                return str(reasoning_content)
            reasoning_details = _get_attr(message, "reasoning_details") or []
            if reasoning_details:
                first_detail = reasoning_details[0]
                return str(_get_attr(first_detail, "text") or "")
    except Exception:
        return ""
    return ""
