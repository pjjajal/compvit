import warnings

### Ignore pesky warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

### Torch Imports
import torch
import torch.nn
import torch.utils.data
import torch.utils.benchmark as bench

### Python STDLIB
import argparse
from typing import Any, List, Tuple, Iterator

### CompViT
from compvit.factory import compvit_factory
from dinov2.factory import dinov2_factory

### Commandline Arguments
def get_argparser() -> argparse.ArgumentParser:
    ### Grab any commandline arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, add_help=False
    )

    parser.add_argument("--model-name", default="compvitg14", choices=['compvitg14', 'compvitb14', 'compvits14', 'compvitl14', "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Specifies the size of the batch_size dimension for created tensors passed through models",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Specifies which device to use for torch operations. Default is cuda",
    )

    parser.add_argument("--no-profile", action="store_true")

    return parser

### Create a benchmark function (very simple)
def benchmark_compvit_milliseconds(x : torch.Tensor, model : torch.nn.Module) -> float:
    ### Do the benchmark!
    t0 = bench.Timer(
        stmt=f"model.forward(x)",
        globals={
            "x": x,
            "model": model
        },
        num_threads=1,
    )

    return t0.blocked_autorange(min_run_time=4.0).median * 1e3

if __name__ == "__main__":
    ### Get args, device
    args = get_argparser().parse_args()
    device = torch.device(args.device)

    ### Parse model name, choose appropriate factory function
    if "compvit" in args.model_name:
        print(f"agx_orin_profile.py: Using compvit factory for {args.model_name}")
        model, config = compvit_factory(model_name=args.model_name)
    elif "dinov2" in args.model_name:
        print(f"agx_orin_profile.py: Using dinov2 factory for {args.model_name}")
        model, config = dinov2_factory(model_name=args.model_name)
    else:
        raise RuntimeError(f"No factory function available for model {args.model_name}")

    ### Load model
    model.to(device).eval()

    ### Create torch.profiler instance
    if args.no_profile:
        torchprofiler = torch.profiler.profile(
            ### Create profiler instance
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=4,
                active=1,
                repeat=0,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                worker_name=f"agx_orin_profile_{args.model_name}_batch_size_{args.batch_size}",
                dir_name="data/",
            ),
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        )
        torchprofiler.start()

    ### Turn off gradient compute
    with torch.no_grad():
        ### Run Benchmark for latency, then do torch profiling!
        rand_x = torch.randn(size=(1, 3, 224, 224), dtype=torch.float32, device=device)

        ### Record latency with benchmark utility
        latency_ms = benchmark_compvit_milliseconds(rand_x, model)
        print("agx_orin_profile.py: Median latency is {:.2f} ms".format(latency_ms))

        if not args.no_profile:
            ### Now do torch.profiler(...)
            print("agx_orin_profile.py: Applying torch.profiler...")

            ### Do some steps
            for k in range(6):
                model(rand_x)
                torchprofiler.step()
                print(f"agx_oring_profile.py: Step {k}")
    
    torchprofiler.stop()
    print("agx_orin_profile.py: Done!")