from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Union, overload

from ai21.stream.async_stream import AsyncStream
from httpx import AsyncClient as AsyncHTTPClient
from typing_extensions import assert_never

from pydantic_ai_slim.pydantic_ai import _utils

from .. import UnexpectedModelBehavior, result
from ..messages import (
    ArgsJson,
    Message,
    ModelAnyResponse,
    ModelStructuredResponse,
    ModelTextResponse,
    RetryPrompt,
    ToolCall,
    ToolReturn,
)
from ..result import Cost
from ..tools import ToolDefinition
from . import (
    AgentModel,
    EitherStreamedResponse,
    Model,
    StreamStructuredResponse,
    StreamTextResponse,
    cached_async_http_client,
    check_allow_model_requests,
)

try:
    from ai21 import NOT_GIVEN, AsyncAI21Client
    from ai21.models import chat
    from ai21.models.chat import ChatCompletionChunk, ChatCompletionResponse
    from ai21.stream.async_stream import AsyncStream
    from ai21.types import NOT_GIVEN
except ImportError as _import_error:
    raise ImportError(
        "Please install `ai21` to use the AI21 model, "
        "you can use the `ai21` optional group — `pip install 'pydantic-ai[ai21]'`"
    ) from _import_error

AI21ModelName = Union[Literal["jamba-1.5-mini", "jamba-1.5-large"], str]
"""
Using this more broad type for the model name instead of the ChatModel definition
allows this model to be used more easily with other model types (ie, Ollama)
"""


@dataclass(init=False)
class AI21Model(Model):
    """A model that uses the AI21 API.

    Internally, this uses the [AI21 Python client](https://github.com/AI21Labs/ai21-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    model_name: AI21ModelName
    client: AsyncAI21Client = field(repr=False)

    def __init__(
        self,
        model_name: AI21ModelName,
        *,
        api_key: str | None = None,
        ai21_client: AsyncAI21Client | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        """Initialize an AI21 model.

        Args:
            model_name: The name of the AI21 model to use.
            api_key: The API key to use for authentication, if not provided, the `AI21_API_KEY` environment variable
                will be used if available.
            ai21_client: An existing
                [`AsyncAI21Client`](https://github.com/AI21Labs/ai21-python/tree/main?tab=readme-ov-file#async-usage)
                client to use, if provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        self.model_name: AI21ModelName = model_name
        if ai21_client is not None:
            assert (
                http_client is None
            ), "Cannot provide both `ai21_client` and `http_client`"
            assert api_key is None, "Cannot provide both `ai21_client` and `api_key`"
            self.client = ai21_client
        elif http_client is not None:
            self.client = AsyncAI21Client(api_key=api_key, http_client=http_client)
        else:
            self.client = AsyncAI21Client(
                api_key=api_key, http_client=cached_async_http_client()
            )

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        check_allow_model_requests()
        tools = [self._map_tool_definition(r) for r in function_tools]
        if result_tools:
            tools += [self._map_tool_definition(r) for r in result_tools]
        return AI21AgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
        )

    def name(self) -> str:
        return f"ai21:{self.model_name}"

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> chat.ToolDefinition:
        return {
            "type": "function",
            "function": {
                "name": f.name,
                "description": f.description,
                "parameters": {
                    "type": "object",
                    "properties": f.parameters_json_schema.get("properties", {}),
                    "required": f.parameters_json_schema.get("required", []),
                },
            },
        }


@dataclass
class AI21AgentModel(AgentModel):
    """Implementation of `AgentModel` for AI21 models."""

    client: AsyncAI21Client
    model_name: AI21ModelName
    allow_text_result: bool
    tools: list[chat.ToolDefinition]

    async def request(
        self, messages: list[Message]
    ) -> tuple[ModelAnyResponse, result.Cost]:
        response = await self._completions_create(messages, False)
        return self._process_response(response), _map_cost(response)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[Message]
    ) -> AsyncIterator[EitherStreamedResponse]:
        response = await self._completions_create(messages, True)
        async with response:
            yield await self._process_streamed_response(response)

    @overload
    async def _completions_create(
        self, messages: list[Message], stream: Literal[True]
    ) -> AsyncStream[ChatCompletionChunk]:
        pass

    @overload
    async def _completions_create(
        self, messages: list[Message], stream: Literal[False]
    ) -> ChatCompletionResponse:
        pass

    async def _completions_create(
        self,
        messages: list[Message],
        stream: bool,
    ) -> ChatCompletionResponse | AsyncStream[ChatCompletionChunk]:
        ai21_messages = [self._map_message(m) for m in messages]
        return await self.client.chat.completions.create(  # type: ignore
            model="jamba-1.5-mini",
            messages=ai21_messages,
            n=1,
            tools=self.tools or NOT_GIVEN,
            stream=stream,
        )

    @staticmethod
    def _process_response(response: chat.ChatCompletionResponse) -> ModelAnyResponse:
        choice = response.choices[0]
        if choice.message.tool_calls is not None:
            return ModelStructuredResponse(
                [
                    ToolCall.from_json(c.function.name, c.function.arguments, c.id)
                    for c in choice.message.tool_calls
                ],
            )
        else:
            assert choice.message.content is not None, choice
            return ModelTextResponse(choice.message.content)

    @staticmethod
    async def _process_streamed_response(
        response: AsyncStream[ChatCompletionChunk],
    ) -> EitherStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        start_cost = Cost()
        # the first chunk may contain enough information so we iterate until we get either `tool_calls` or `content`
        while True:
            try:
                chunk = await response.__anext__()
            except StopAsyncIteration as e:
                raise UnexpectedModelBehavior(
                    "Streamed response ended without content or tool calls"
                ) from e

            start_cost += _map_cost(chunk)

            if chunk.choices:
                delta = chunk.choices[0].delta

                if delta.content is not None:
                    return AI21StreamTextResponse(
                        delta.content, response, _utils.now_utc(), start_cost
                    )
                # else continue until we get either delta.content or delta.tool_calls (tool_calls not supported yet)

    @staticmethod
    def _map_message(message: Message) -> chat.ChatMessage:
        if message.role == "system":
            # SystemPrompt ->
            return chat.SystemMessage(content=message.content)
        elif message.role == "user":
            # UserPrompt ->
            return chat.UserMessage(content=message.content)
        elif message.role == "tool-return":
            # ToolReturn ->
            return chat.ToolMessage(
                role="tool",
                tool_call_id=_guard_tool_call_id(message),
                content=message.model_response_str(),
            )
        elif message.role == "retry-prompt":
            # RetryPrompt ->
            if message.tool_name is None:
                return chat.UserMessage(content=message.model_response())
            else:
                return chat.ToolMessage(
                    tool_call_id=_guard_tool_call_id(message),
                    content=message.model_response(),
                )
        elif message.role == "model-text-response":
            # ModelTextResponse ->
            return chat.AssistantMessage(content=message.content)
        elif message.role == "model-structured-response":
            assert (
                message.role == "model-structured-response"
            ), f'Expected role to be "llm-tool-calls", got {message.role}'
            # ModelStructuredResponse ->
            return chat.AssistantMessage(
                tool_calls=[_map_tool_call(t) for t in message.calls],
            )
        else:
            assert_never(message)


@dataclass
class AI21StreamTextResponse(StreamTextResponse):
    """Implementation of `StreamTextResponse` for AI21 models."""

    _first: str | None
    _response: AsyncStream[ChatCompletionChunk]
    _timestamp: datetime
    _cost: result.Cost
    _buffer: list[str] = field(default_factory=list, init=False)

    async def __anext__(self) -> None:
        if self._first is not None:
            self._buffer.append(self._first)
            self._first = None
            return None

        chunk = await self._response.__anext__()
        self._cost += _map_cost(chunk)
        try:
            choice = chunk.choices[0]
        except IndexError:
            raise StopAsyncIteration()

        # we don't raise StopAsyncIteration on the last chunk because usage comes after this
        if choice.finish_reason is None:
            assert (
                choice.delta.content is not None
            ), f"Expected delta with content, invalid chunk: {chunk!r}"
        if choice.delta.content is not None:
            self._buffer.append(choice.delta.content)

    def get(self, *, final: bool = False) -> Iterable[str]:
        yield from self._buffer
        self._buffer.clear()

    def cost(self) -> Cost:
        return self._cost

    def timestamp(self) -> datetime:
        return self._timestamp


@dataclass
class AI21StreamStructuredResponse(StreamStructuredResponse):
    """Implementation of `StreamStructuredResponse` for AI21 models."""

    _response: AsyncStream[ChatCompletionChunk]
    _delta_tool_calls: dict[int, chat.ChoiceDelta]
    _timestamp: datetime
    _cost: result.Cost

    async def __anext__(self) -> None:
        chunk = await self._response.__anext__()
        self._cost += _map_cost(chunk)
        try:
            choice = chunk.choices[0]
        except IndexError:
            raise StopAsyncIteration()

        if choice.finish_reason is not None:
            raise StopAsyncIteration()

    def get(self, *, final: bool = False) -> ModelStructuredResponse:
        calls: list[ToolCall] = []

        return ModelStructuredResponse(calls, timestamp=self._timestamp)

    def cost(self) -> Cost:
        return self._cost

    def timestamp(self) -> datetime:
        return self._timestamp


def _guard_tool_call_id(t: ToolCall | ToolReturn | RetryPrompt) -> str:
    """Type guard that checks a `tool_call_id` is not None both for static typing and runtime."""
    assert t.tool_call_id is not None, f"AI21 requires `tool_call_id` to be set: {t}"
    return t.tool_call_id


def _map_tool_call(t: ToolCall) -> chat.ToolCall:
    assert isinstance(t.args, ArgsJson), f"Expected ArgsJson, got {t.args}"
    return chat.ToolCall(
        id=_guard_tool_call_id(t),
        type="function",
        function=chat.ToolFunction(
            name=t.tool_name,
            arguments=t.args.args_json,
        ),
    )


def _map_cost(
    response: ChatCompletionResponse | ChatCompletionChunk,
) -> result.Cost:
    usage = response.usage
    if usage is None:
        return result.Cost()
    else:
        details: dict[str, int] = {}

        return result.Cost(
            request_tokens=usage.prompt_tokens,
            response_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            details=details,
        )
