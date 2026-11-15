# Adapted for OrbitInfer; see NOTICE and docs/source-import.md.
"""Chat with the LFM2 Torch runner using OrbitInfer's native conversation recipe."""

import torch
from tokenizers import Tokenizer

import orbitinfer
from orbitinfer import Engine, SamplingParams
from orbitinfer.recipe import get_native_recipe
from orbitinfer_models_torch.lfm2 import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    Lfm2TorchRunner,
)


def main() -> None:
    torch.set_num_threads(1)

    model = DEFAULT_MODEL
    tokenizer = Tokenizer.from_pretrained(model, revision=DEFAULT_REVISION)
    recipe = get_native_recipe("lfm2", tokenizer)
    conversation = [
        orbitinfer.recipe.Message(
            orbitinfer.recipe.Role.SYSTEM,
            [
                orbitinfer.recipe.Text(
                    "You are a helpful assistant trained by Liquid AI."
                )
            ],
        ),
        orbitinfer.recipe.Message(
            orbitinfer.recipe.Role.USER,
            [orbitinfer.recipe.Text("What is the capital of France?")],
        ),
    ]
    prompt = recipe.render(conversation)
    parser = recipe.create_parser(prompt)

    runner = Lfm2TorchRunner(model, device="cpu")
    with Engine(runner) as engine:
        request = engine.submit(
            prompt.token_ids,
            sampling_params=SamplingParams(
                max_output_tokens=256, stop_token_ids=recipe.stop_token_ids
            ),
        )
        while engine.has_work():
            for token in engine.step():
                events = parser.feed([token.token_id])
                if token.finished:
                    reason = token.finish_reason
                    if reason == "stop":
                        events.extend(parser.finish("stop"))
                    elif reason == "length":
                        events.extend(parser.finish("length"))
                    else:
                        raise RuntimeError(f"unexpected completion reason: {reason!r}")
                for event in events:
                    if isinstance(event, orbitinfer.recipe.TextDelta):
                        print(event.delta, end="", flush=True)
        request.take_result()
    print()


if __name__ == "__main__":
    main()
