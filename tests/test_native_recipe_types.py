# Adapted for OrbitInfer; see NOTICE and docs/source-import.md.
"""Ownership and state boundaries of the native conversation interface."""

from __future__ import annotations

import gc
import json
import weakref

import numpy as np
import pytest
from output_helpers import forward_output
from test_http_lifecycle import post, running_server
from test_recipe import _byte_tokenizer

import orbitinfer
from orbitinfer.recipe import InklingMedia, NativeRecipe, get_native_recipe


def tokenizer():
    backend = _byte_tokenizer()
    backend.add_special_tokens(
        [
            "<|start|>",
            "<|end|>",
            "<|return|>",
            "<|call|>",
            "<|channel|>",
            "<|message|>",
            "<|constrain|>",
            "<|message_user|>",
            "<|message_system|>",
            "<|message_tool|>",
            "<|content_image|>",
            "<|unused_200054|>",
        ]
    )
    return backend


def test_typed_inputs_own_json_values_and_copy_out_fields():
    parameters = {"type": "object", "properties": {"n": {"type": "integer"}}}
    tool = orbitinfer.recipe.Tool("lookup", parameters=parameters)
    call = orbitinfer.recipe.ToolCall("lookup", {"n": 3}, id="call-1")
    message = orbitinfer.recipe.Message(orbitinfer.recipe.Role.ASSISTANT, [call])
    options = orbitinfer.recipe.RenderOptions(response_format={"type": "json_object"})
    parameters["properties"]["n"]["type"] = "string"
    tool.parameters["properties"]["n"]["type"] = "boolean"
    call.arguments["n"] = 99
    options.response_format["type"] = "text"
    assert tool.parameters["properties"]["n"]["type"] == "integer"
    saved_call = message.content[0]
    assert isinstance(saved_call, orbitinfer.recipe.ToolCall)
    assert saved_call.arguments == {"n": 3}
    assert options.response_format == {"type": "json_object"}
    assert isinstance(message.content, tuple)
    with pytest.raises(AttributeError):
        setattr(message, "role", orbitinfer.recipe.Role.USER)
    with pytest.raises(AttributeError):
        setattr(saved_call, "name", "changed")


def test_native_prompt_is_owned_bound_to_one_recipe_and_parsers_are_fresh():
    backend = tokenizer()
    first = get_native_recipe("gpt_oss", backend)
    second = get_native_recipe("gpt_oss", backend)
    prompt = first.render(
        [
            orbitinfer.recipe.Message(
                orbitinfer.recipe.Role.USER, [orbitinfer.recipe.Text("hi")]
            )
        ]
    )
    original = prompt.token_ids
    assert np.shares_memory(prompt.token_ids, original)
    with pytest.raises(ValueError, match="read-only"):
        original[0] = 99
    assert first.stop_token_ids
    with pytest.raises(ValueError, match="this recipe"):
        second.create_parser(prompt)
    with pytest.raises(ValueError, match="this recipe"):
        first.create_parser(orbitinfer.recipe.RenderedPrompt(original))
    ids = backend.encode("thought", add_special_tokens=False).ids
    a, b = first.create_parser(prompt), first.create_parser(prompt)
    events = a.feed(ids)
    assert events == b.feed(ids)
    assert events
    assert all(isinstance(event, orbitinfer.recipe.ReasoningDelta) for event in events)
    assert all(isinstance(event, orbitinfer.recipe.ParserEvent) for event in events)
    assert a is not b
    assert first.parse_response(ids, prompt=prompt)["reasoning_content"] == "thought"


def test_native_events_preserve_variants_values_and_owned_provenance():
    spans = [(0, 0, 2, 0, 2)]
    text = orbitinfer.recipe.TextDelta("é", token_spans=spans)
    reasoning = orbitinfer.recipe.ReasoningDelta("é", token_spans=spans)
    start = orbitinfer.recipe.ToolCallStart(2, "call-2", "lookup", token_spans=spans)
    arguments = orbitinfer.recipe.ToolCallArgumentsDelta(2, '{"x":', token_spans=spans)
    spans.clear()
    assert text.type == "text_delta" and text.delta == "é"
    assert reasoning.type == "reasoning_delta" and reasoning.delta == "é"
    assert start.type == "tool_call_start"
    assert (start.index, start.id, start.name) == (2, "call-2", "lookup")
    assert arguments.type == "tool_call_arguments_delta"
    assert (arguments.index, arguments.delta) == (2, '{"x":')
    assert text == orbitinfer.recipe.TextDelta("é", token_spans=[(0, 0, 2, 0, 2)])
    assert text != reasoning
    assert text != orbitinfer.recipe.TextDelta("é")
    assert text != {"type": "text_delta", "delta": "é"}
    for event in (text, reasoning, start, arguments):
        assert isinstance(event, orbitinfer.recipe.ParserEvent)
        assert event.token_spans == ((0, 0, 2, 0, 2),)
        assert isinstance(event.token_spans, tuple)
        with pytest.raises(AttributeError):
            setattr(event, "token_spans", ())
    with pytest.raises(AttributeError):
        setattr(text, "delta", "changed")
    with pytest.raises(AttributeError):
        setattr(start, "index", 0)
    assert orbitinfer.recipe.TextDelta("synthetic").token_spans == ()


@pytest.mark.parametrize(
    ("spans", "error"),
    [
        ([(0, 0, 1, 0)], ValueError),
        ([(-1, 0, 1, 0, 1)], OverflowError),
        ([(1 << 32, 0, 1, 0, 1)], OverflowError),
    ],
)
def test_native_event_constructor_validates_span_storage(spans, error):
    with pytest.raises(error):
        orbitinfer.recipe.TextDelta("x", token_spans=spans)


def test_parser_does_not_retain_media_or_the_rendered_prompt():
    native = get_native_recipe("inkling", tokenizer())
    pixels = np.zeros((40, 40, 3), dtype=np.float32)
    source = weakref.ref(pixels)
    messages = [
        orbitinfer.recipe.Message(
            orbitinfer.recipe.Role.USER, [orbitinfer.recipe.Image(source=pixels)]
        )
    ]
    prompt = native.render(messages)
    assert isinstance(prompt.media, InklingMedia)
    assert prompt.media.pixel_values is not None
    reference = weakref.ref(prompt.media.pixel_values)
    parser = native.create_parser(prompt)
    del pixels, messages, prompt
    gc.collect()
    assert source() is None
    assert reference() is None
    assert parser.feed([]) == []


def test_native_media_accounts_for_each_source():
    native = get_native_recipe("inkling", tokenizer())
    valid = orbitinfer.recipe.Image(source=np.zeros((40, 40, 3), dtype=np.float32))
    prompt = native.render(
        [orbitinfer.recipe.Message(orbitinfer.recipe.Role.USER, [valid, valid])]
    )
    assert isinstance(prompt.media, InklingMedia)
    assert prompt.media.pixel_values is not None
    assert prompt.media.pixel_values.shape[0] == 4
    assert len(prompt.media.image_positions) == 4
    with pytest.raises(ValueError, match="dtype float32"):
        native.render(
            [
                orbitinfer.recipe.Message(
                    orbitinfer.recipe.Role.USER,
                    [valid, orbitinfer.recipe.Image(source=object())],
                )
            ]
        )


def test_public_order_and_per_message_controls_are_not_silently_dropped():
    native = get_native_recipe("gpt_oss", tokenizer())
    messages = [
        orbitinfer.recipe.Message(
            orbitinfer.recipe.Role.ASSISTANT,
            [
                orbitinfer.recipe.ToolCall("f", {}),
                orbitinfer.recipe.Text("later text"),
            ],
        )
    ]
    with pytest.raises(ValueError, match="block order"):
        native.debug_render_text(messages)
    with pytest.raises(ValueError, match="per-message"):
        native.debug_render_text(
            [
                orbitinfer.recipe.Message(
                    orbitinfer.recipe.Role.USER,
                    [orbitinfer.recipe.Text("hi")],
                    task="action",
                )
            ]
        )
    with pytest.raises(ValueError, match="initial"):
        native.debug_render_text(
            [
                orbitinfer.recipe.Message(
                    orbitinfer.recipe.Role.USER, [orbitinfer.recipe.Text("hi")]
                ),
                orbitinfer.recipe.Message(
                    orbitinfer.recipe.Role.SYSTEM,
                    [],
                    tools=[orbitinfer.recipe.Tool("f")],
                ),
            ]
        )
    with pytest.raises(ValueError, match="does not support rendering option"):
        native.debug_render_text(
            [
                orbitinfer.recipe.Message(
                    orbitinfer.recipe.Role.USER, [orbitinfer.recipe.Text("hi")]
                )
            ],
            options=orbitinfer.recipe.RenderOptions(add_vision_id=True),
        )
    with pytest.raises(ValueError, match="latest_reminder"):
        native.debug_render_text(
            [
                orbitinfer.recipe.Message(
                    orbitinfer.recipe.Role.LATEST_REMINDER,
                    [orbitinfer.recipe.Text("Remember")],
                )
            ]
        )


def test_raw_tool_arguments_survive_without_json_reserialization():
    native = get_native_recipe(
        "gpt_oss",
        tokenizer(),
        use_system_prompt=False,
    )
    raw = '{ "n": 1.00, "nested": [true, null] }'
    text = native.debug_render_text(
        [
            orbitinfer.recipe.Message(
                orbitinfer.recipe.Role.USER, [orbitinfer.recipe.Text("run")]
            ),
            orbitinfer.recipe.Message(
                orbitinfer.recipe.Role.ASSISTANT, [orbitinfer.recipe.ToolCall("f", raw)]
            ),
        ]
    )
    assert raw in text


def test_native_http_lowering_matches_public_typed_messages():
    backend = tokenizer()
    backend.add_special_tokens(
        [
            "<bos>",
            "<|turn>",
            "<turn|>",
            "<eos>",
            "<|channel>",
            "<channel|>",
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
            '<|"|>',
        ]
    )
    native = get_native_recipe("gemma4", backend)
    messages = [
        orbitinfer.recipe.Message(
            orbitinfer.recipe.Role.USER, [orbitinfer.recipe.Text("Go")]
        ),
        orbitinfer.recipe.Message(
            orbitinfer.recipe.Role.ASSISTANT,
            [
                orbitinfer.recipe.Thinking("Use the tool"),
                orbitinfer.recipe.ToolCall("f", {"n": 3}, id="c1"),
                orbitinfer.recipe.ToolResult({"answer": 4}, name="f"),
                orbitinfer.recipe.Text("Here"),
            ],
        ),
        orbitinfer.recipe.Message(
            orbitinfer.recipe.Role.USER, [orbitinfer.recipe.Text("Next")]
        ),
    ]
    reconstructed = [
        orbitinfer.recipe.Message(message.role, message.content) for message in messages
    ]
    expected = native.render(messages).token_ids
    np.testing.assert_array_equal(native.render(reconstructed).token_ids, expected)
    seen = []

    class Runner:
        def forward(self, plan):
            seen.extend(plan.token_ids.tolist())
            return forward_output([backend.token_to_id("<eos>")])

    wire_messages = [
        {"role": "user", "content": "Go"},
        {
            "role": "assistant",
            "content": "Here",
            "reasoning_content": "Use the tool",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": {"n": 3}},
                }
            ],
            "tool_responses": [{"name": "f", "response": {"answer": 4}}],
        },
        {"role": "user", "content": "Next"},
    ]
    with running_server(Runner(), native) as (_, _, _, errors, base, _):
        with post(
            base,
            json.dumps(
                {"model": "test", "messages": wire_messages, "max_tokens": 1}
            ).encode(),
        ) as response:
            assert response.status == 200
        assert not errors
    np.testing.assert_array_equal(seen, expected)


def test_rendered_prompt_token_view_is_owned_readonly_and_outlives_prompt():
    source = np.asarray([11, 22, 33], dtype=np.uint32)
    prompt = orbitinfer.recipe.RenderedPrompt(source)
    source[0] = 99
    tokens = prompt.token_ids
    assert tokens.dtype == np.uint32 and tokens.ndim == 1
    assert tokens.flags.c_contiguous and not tokens.flags.writeable
    assert tokens.base is prompt
    assert np.shares_memory(tokens, prompt.token_ids)
    np.testing.assert_array_equal(tokens, [11, 22, 33])
    with pytest.raises(ValueError, match="read-only"):
        tokens[0] = 7
    with pytest.raises(ValueError, match="WRITEABLE"):
        tokens.flags.writeable = True
    del prompt
    gc.collect()
    np.testing.assert_array_equal(tokens, [11, 22, 33])
    empty = orbitinfer.recipe.RenderedPrompt([]).token_ids
    assert empty.dtype == np.uint32 and empty.shape == (0,)
    assert not empty.flags.writeable


@pytest.mark.parametrize(
    ("recipe_name", "default_markers"),
    [
        ("gpt_oss", ["<|return|>", "<|call|>"]),
        ("deepseek_v41", ["<｜end▁of▁sentence｜>"]),
        ("inkling", ["<|content_model_end_sampling|>"]),
        ("qwen3_5", ["<|im_end|>", "<|endoftext|>"]),
        ("qwen3_8", ["<|im_end|>", "<|endoftext|>"]),
        ("gemma4", ["<turn|>", "<eos>"]),
        ("lfm2", ["<|im_end|>"]),
    ],
)
def test_native_recipe_stop_defaults_and_overrides_share_one_contract(
    recipe_name, default_markers
):
    backend = tokenizer()
    backend.add_special_tokens(default_markers)
    expected = [backend.token_to_id(marker) for marker in default_markers]
    default = get_native_recipe(recipe_name, backend)
    assert type(default) is NativeRecipe and default.tokenizer.is_native
    assert isinstance(default.stop_token_ids, list)
    assert default.stop_token_ids == expected
    for override, wanted in [(None, expected), ([], [])]:
        assert (
            get_native_recipe(
                recipe_name, backend, stop_token_ids=override
            ).stop_token_ids
            == wanted
        )

    source = [901, 902]
    recipe = get_native_recipe(recipe_name, backend, stop_token_ids=source)
    source[0] = 999
    stops = recipe.stop_token_ids
    assert isinstance(stops, list)
    assert stops == [901, 902]
    stops[0] = 0
    stops.append(903)
    assert recipe.stop_token_ids == [901, 902]
