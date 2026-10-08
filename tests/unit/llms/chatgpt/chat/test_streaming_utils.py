"""
Tests for ChatGPTToolCallNormalizer.

Verifies that non-spec-compliant tool_call chunks from the ChatGPT backend API
are normalized to match the OpenAI streaming spec:
- Correct index assignment for parallel tool calls
- Deduplication of "closing" chunks with repeated id/name
"""

import pytest

from litellm.llms.chatgpt.chat.streaming_utils import ChatGPTToolCallNormalizer
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)


def _make_chunk(tool_calls=None, content=None):
    """Helper to build a ModelResponseStream chunk with tool_calls on the delta."""
    delta = Delta(
        content=content,
        role="assistant",
        tool_calls=tool_calls,
    )
    choice = StreamingChoices(delta=delta, index=0)
    return ModelResponseStream(choices=[choice])


def _make_tc(index=0, id=None, name=None, arguments=None):
    """Helper to build a ChatCompletionDeltaToolCall."""
    func = Function(name=name, arguments=arguments)
    return ChatCompletionDeltaToolCall(
        index=index,
        id=id,
        function=func,
        type="function" if id else None,
    )


class TestChatGPTToolCallNormalizer:
    """Test that the normalizer fixes ChatGPT-style tool_call streaming issues."""

    def test_single_tool_call_index_preserved(self):
        """A single tool call should get index=0."""
        chunks = [
            _make_chunk(
                tool_calls=[_make_tc(index=0, id="call_1", name="get_weather")]
            ),
            _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"loc')]),
            _make_chunk(tool_calls=[_make_tc(index=0, arguments='ation": "NYC"}')]),
        ]
        normalizer = ChatGPTToolCallNormalizer(iter(chunks))
        results = list(normalizer)

        assert len(results) == 3
        assert results[0].choices[0].delta.tool_calls[0].index == 0
        assert results[0].choices[0].delta.tool_calls[0].id == "call_1"
        assert results[1].choices[0].delta.tool_calls[0].index == 0
        assert results[2].choices[0].delta.tool_calls[0].index == 0

    def test_parallel_tool_calls_get_correct_indices(self):
        """
        ChatGPT sends all tool_calls with index=0. The normalizer should assign
        sequential indices: 0 for the first, 1 for the second.
        """
        chunks = [
            # First tool call: intro chunk with id + name
            _make_chunk(
                tool_calls=[_make_tc(index=0, id="call_aaa", name="get_weather")]
            ),
            # First tool call: arguments streaming
            _make_chunk(
                tool_calls=[_make_tc(index=0, arguments='{"location": "NYC"}')]
            ),
            # First tool call: duplicate closing chunk (id repeated) — should be skipped
            _make_chunk(
                tool_calls=[_make_tc(index=0, id="call_aaa", name="get_weather")]
            ),
            # Second tool call: intro chunk with id + name (index=0 from ChatGPT)
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_bbb", name="get_time")]),
            # Second tool call: arguments streaming
            _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"tz": "EST"}')]),
            # Second tool call: duplicate closing chunk — should be skipped
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_bbb", name="get_time")]),
        ]

        normalizer = ChatGPTToolCallNormalizer(iter(chunks))
        results = list(normalizer)

        # 2 duplicate chunks should be skipped → 4 results
        assert len(results) == 4

        # First tool call chunks should have index=0
        assert results[0].choices[0].delta.tool_calls[0].index == 0
        assert results[0].choices[0].delta.tool_calls[0].id == "call_aaa"
        assert results[1].choices[0].delta.tool_calls[0].index == 0

        # Second tool call chunks should have index=1
        assert results[2].choices[0].delta.tool_calls[0].index == 1
        assert results[2].choices[0].delta.tool_calls[0].id == "call_bbb"
        assert results[3].choices[0].delta.tool_calls[0].index == 1

    def test_non_tool_call_chunks_pass_through(self):
        """Chunks without tool_calls should pass through unchanged."""
        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(content=" world"),
        ]
        normalizer = ChatGPTToolCallNormalizer(iter(chunks))
        results = list(normalizer)

        assert len(results) == 2
        assert results[0].choices[0].delta.content == "Hello"
        assert results[1].choices[0].delta.content == " world"

    def test_empty_choices_pass_through(self):
        """Chunks with empty choices should pass through."""
        chunk = ModelResponseStream(choices=[])
        normalizer = ChatGPTToolCallNormalizer(iter([chunk]))
        results = list(normalizer)

        assert len(results) == 1

    def test_three_parallel_tool_calls(self):
        """Three parallel tool calls should get indices 0, 1, 2."""
        chunks = [
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_1", name="fn_a")]),
            _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"a":1}')]),
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_2", name="fn_b")]),
            _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"b":2}')]),
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_3", name="fn_c")]),
            _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"c":3}')]),
        ]

        normalizer = ChatGPTToolCallNormalizer(iter(chunks))
        results = list(normalizer)

        assert len(results) == 6
        # First tool call
        assert results[0].choices[0].delta.tool_calls[0].index == 0
        assert results[1].choices[0].delta.tool_calls[0].index == 0
        # Second tool call
        assert results[2].choices[0].delta.tool_calls[0].index == 1
        assert results[3].choices[0].delta.tool_calls[0].index == 1
        # Third tool call
        assert results[4].choices[0].delta.tool_calls[0].index == 2
        assert results[5].choices[0].delta.tool_calls[0].index == 2

    def test_all_duplicates_skipped(self):
        """If a chunk contains only duplicate tool_calls, the entire chunk is skipped."""
        chunks = [
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_x", name="fn")]),
            # Duplicate — same id seen before
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_x", name="fn")]),
        ]

        normalizer = ChatGPTToolCallNormalizer(iter(chunks))
        results = list(normalizer)

        assert len(results) == 1
        assert results[0].choices[0].delta.tool_calls[0].id == "call_x"

    @pytest.mark.asyncio
    async def test_async_iteration(self):
        """The normalizer should work with async iteration."""

        async def async_gen():
            chunks = [
                _make_chunk(tool_calls=[_make_tc(index=0, id="call_a", name="fn_a")]),
                _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"x":1}')]),
                _make_chunk(tool_calls=[_make_tc(index=0, id="call_b", name="fn_b")]),
                _make_chunk(tool_calls=[_make_tc(index=0, arguments='{"y":2}')]),
            ]
            for c in chunks:
                yield c

        normalizer = ChatGPTToolCallNormalizer(async_gen())
        results = []
        async for chunk in normalizer:
            results.append(chunk)

        assert len(results) == 4
        assert results[0].choices[0].delta.tool_calls[0].index == 0
        assert results[2].choices[0].delta.tool_calls[0].index == 1

    def test_getattr_proxies_to_stream(self):
        """Attribute access should be proxied to the underlying stream."""

        class FakeStream:
            custom_attr = "test_value"

            def __iter__(self):
                return iter([])

            def __next__(self):
                raise StopIteration

        normalizer = ChatGPTToolCallNormalizer(FakeStream())
        assert normalizer.custom_attr == "test_value"

    def test_done_derived_chunk_with_seen_id_and_arguments_is_kept(self):
        """A chunk that reuses an already-seen tool call id but carries
        arguments comes from response.function_call_arguments.done (the full
        arguments for a call that never streamed deltas). It must be kept and
        attributed to its own call, not dropped as a duplicate closing chunk."""
        chunks = [
            # Codex order: every call is introduced before any arguments arrive
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_1", name="get_weather")]),
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_2", name="get_weather")]),
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_1", arguments='{"city": "Paris"}')]),
            _make_chunk(tool_calls=[_make_tc(index=0, id="call_2", arguments='{"city": "London"}')]),
        ]

        normalizer = ChatGPTToolCallNormalizer(iter(chunks))
        results = list(normalizer)

        assert len(results) == 4
        first_args = results[2].choices[0].delta.tool_calls[0]
        second_args = results[3].choices[0].delta.tool_calls[0]
        assert (first_args.index, first_args.function.arguments) == (0, '{"city": "Paris"}')
        assert (second_args.index, second_args.function.arguments) == (1, '{"city": "London"}')

    def test_codex_parallel_tool_calls_end_to_end(self):
        """Raw Codex backend events (no argument deltas) flow through the
        Responses->chat bridge and the normalizer; each parallel call ends up
        with its own arguments on its own index."""
        from litellm.completion_extras.litellm_responses_transformation.transformation import (
            OpenAiResponsesToChatCompletionStreamIterator,
        )

        events = [
            {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}},
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {"type": "function_call", "id": "fc_1", "call_id": "call_aaa", "name": "get_weather"},
            },
            {
                "type": "response.output_item.added",
                "output_index": 2,
                "item": {"type": "function_call", "id": "fc_2", "call_id": "call_bbb", "name": "get_weather"},
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "output_index": 1,
                "arguments": '{"city": "Paris"}',
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_2",
                "output_index": 2,
                "arguments": '{"city": "London"}',
            },
        ]

        bridge = OpenAiResponsesToChatCompletionStreamIterator(streaming_response=None, sync_stream=True)
        normalizer = ChatGPTToolCallNormalizer(iter([bridge.chunk_parser(event) for event in events]))
        results = list(normalizer)

        per_index_arguments = {}
        for result in results:
            for tool_call in result.choices[0].delta.tool_calls or []:
                arguments = tool_call.function.arguments if tool_call.function else ""
                if arguments:
                    per_index_arguments[tool_call.index] = arguments

        assert per_index_arguments == {0: '{"city": "Paris"}', 1: '{"city": "London"}'}
