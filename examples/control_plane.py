"""CPU control-plane example with checked tokens; this is not a language model."""

import numpy as np

from orbitinfer import Engine, ForwardOutput, SamplingParams


class CountingRunner:
    """Exercise the real engine without GPU computation or external storage."""

    def forward(self, plan):
        tokens = np.asarray(
            [int(plan.token_ids[end - 1]) + 1 for end in plan.token_indptr[1:]],
            dtype=np.uint32,
        )
        return ForwardOutput(
            token_ids=tokens,
            token_indptr=np.arange(len(tokens) + 1, dtype=np.uint32),
        )


def main():
    with Engine(CountingRunner(), max_running_requests=2, enable_overlap=False) as engine:
        requests = [
            engine.submit([start], sampling_params=SamplingParams(max_output_tokens=3))
            for start in (10, 20)
        ]
        while engine.has_work():
            engine.step()
        actual = [request.take_result() for request in requests]
        expected = [[11, 12, 13], [21, 22, 23]]
        if actual != expected:
            raise RuntimeError(f"token mismatch: {actual!r} != {expected!r}")
    print("CPU control plane passed:", actual)


if __name__ == "__main__":
    main()
