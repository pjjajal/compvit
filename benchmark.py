import warnings

### Ignore pesky warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

import argparse
from typing import Any, Iterator, List, Tuple

import torch
import torch.utils.benchmark as bench
from torch.profiler import ProfilerActivity, profile, record_function

from compvit.factory import compvit_factory
from dinov2.factory import dinov2_factory


def parse_args():
    parser = argparse.ArgumentParser("Benchmarking Code")
    parser.add_argument(
        "--model",
        choices=[
            "dinov2_vittiny14",
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
            "compvits14",
            "compvitb14",
            "compvitl14",
            "compvitg14",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"])

    return parser.parse_args()


### Create a benchmark function (very simple)
def benchmark_compvit_milliseconds(x: torch.Tensor, model: torch.nn.Module) -> Any:
    ### Do the benchmark!
    t0 = bench.Timer(
        stmt=f"model.forward(x)",
        globals={"x": x, "model": model},
        num_threads=1,
    )

    return t0.blocked_autorange(min_run_time=8.0)


def main(args):
    ### Get args, device
    device = torch.device(args.device)

    ### Parse model name, choose appropriate factory function
    if "compvit" in args.model:
        print(f"Using compvit factory for {args.model}")
        model, config = compvit_factory(model_name=args.model)
    elif "dinov2" in args.model:
        print(f"Using dinov2 factory for {args.model}")
        model, config = dinov2_factory(model_name=args.model)
    else:
        raise RuntimeError(f"No factory function available for model {args.model}")

    ### Load model
    model.to(device).eval()
    print(f"# of parameters: {sum(p.numel() for p in model.parameters()):_}")
    ### Turn off gradient compute
    with torch.no_grad():
        ### Run Benchmark for latency, then do torch profiling!
        rand_x = torch.randn(
            size=(args.batch_size, 3, 224, 224), dtype=torch.float32, device=device
        )

        ### Record latency with benchmark utility
        latency_measurement = benchmark_compvit_milliseconds(rand_x, model)
        latency_mean = latency_measurement.mean * 1e3
        latency_median = latency_measurement.median * 1e3
        latency_iqr = latency_measurement.iqr * 1e3

        print(
            f"{args.model}| Mean/Median/IQR latency (ms) is {latency_mean:.2f} | {latency_median:.2f} | {latency_iqr:.2f}"
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
