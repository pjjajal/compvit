import warnings

### Ignore pesky warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

import argparse
import itertools
import textwrap
from typing import Any, Literal, List, Dict

import torch
import torch.utils.benchmark as bench
import pandas as pd

from compvit.factory import compvit_factory
from dinov2.factory import dinov2_factory


def parse_args():
    parser = argparse.ArgumentParser("Benchmarking Code")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"])

    single_group = parser.add_argument_group(
        title="Benchmark Single",
        description="Use these arguements to benchmark a SINGLE model.",
    )
    single_group.add_argument(
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

    benchmark_all_group = parser.add_argument_group(
        title="Benchmark All",
        description="Use these flags to benchmark a class of models.",
    )
    benchmark_all_group.add_argument("--all-compvit", action="store_true")
    benchmark_all_group.add_argument("--all-dino", action="store_true")
    benchmark_all_group.add_argument("--filetag", default="", type=str)

    sweep_group = parser.add_argument_group(
        title="CompViT Sweep", description="Use these arguments to ablate CompViT."
    )
    sweep_group.add_argument(
        "--sweep-model",
        choices=[
            "compvits14",
            "compvitb14",
            "compvitl14",
            "compvitg14",
        ],
    )
    sweep_group.add_argument("--compvit-sweep", action="store_true")
    sweep_group.add_argument(
        "--compressed-tokens-sweep", nargs="+", type=int, dest="token_sweep"
    )
    sweep_group.add_argument(
        "--bottleneck-loc-sweep", nargs=2, type=int, dest="bottleneck_locs"
    )

    return parser.parse_args()


def colour_text(
    text,
    colour_code: Literal[
        "black",
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "white",
        "reset",
    ],
    *args,
    **kwargs,
):
    colour_codes = {
        "black": "\033[90m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }

    coloured_text = colour_codes[colour_code] + str(text) + colour_codes["reset"]
    return coloured_text


def device_info(args):
    device = torch.device(args.device)
    device_name = ""
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    return device_name


def export_sweep_data(data: List[Dict[str, Any]], filename):
    pd.DataFrame(data).to_csv(filename)


### Create a benchmark function (very simple)
def benchmark_compvit_milliseconds(x: torch.Tensor, model: torch.nn.Module) -> Any:
    ### Do the benchmark!
    t0 = bench.Timer(
        stmt=f"model.forward(x)",
        globals={"x": x, "model": model},
        num_threads=1,
    )

    return t0.blocked_autorange(min_run_time=8.0)


def inference(model, device, batch_size):
    ### Turn off gradient compute
    with torch.no_grad():
        ### Run Benchmark for latency, then do torch profiling!
        rand_x = torch.randn(
            size=(batch_size, 3, 224, 224), dtype=torch.float32, device=device
        )

        ### Record latency with benchmark utility
        latency_measurement = benchmark_compvit_milliseconds(rand_x, model)
        latency_mean = latency_measurement.mean * 1e3
        latency_median = latency_measurement.median * 1e3
        latency_iqr = latency_measurement.iqr * 1e3

    return latency_mean, latency_median, latency_iqr


def test_dino(args):
    dino_models = [
        "dinov2_vits14",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
    ]

    ### Get args, device
    device = torch.device(args.device)

    all_data = []

    # Measure dino models.
    for model_name in dino_models:
        model, config = dinov2_factory(model_name=model_name)

        model = model.to(device).eval()
        latency_mean, latency_median, latency_iqr = inference(
            model, device, args.batch_size
        )

        message = f"""\
        ========================
        {colour_text(model_name.upper(), 'green')}
        {colour_text("Parameters", 'cyan')}: {sum(p.numel() for p in model.parameters()):,}
        {colour_text("Depth", 'cyan')}: {config['depth']}
        {colour_text("Embedding Dim", 'cyan')}: {config['embed_dim']}
        {colour_text("Mean (ms)", "magenta")}: {latency_mean:.2f} 
        {colour_text("Median (ms)", "magenta")}: {latency_median:.2f}
        {colour_text("IQR (ms)", "magenta")}: {latency_iqr:.2f}
        ========================\
        """
        message = textwrap.dedent(message)

        print(message)

        all_data.append(
            {
                "Parameters": sum(p.numel() for p in model.parameters()),
                "Depth": config["depth"],
                "Embedding Dim": config["embed_dim"],
                "Mean (ms)": latency_mean,
                "Median (ms)": latency_median,
                "IQR (ms)": latency_iqr,
            }
        )

    filename = (
        "_".join(
            [
                device_info(args).replace(" ", ""),
                "dinov2",
                f"bs{args.batch_size}",
                f"{args.filetag}",
            ]
        )
        + ".csv"
    )
    export_sweep_data(all_data, filename)


def test_compvit(args):
    compvit_models = [
        "compvits14",
        "compvitb14",
        "compvitl14",
        "compvitg14",
    ]

    ### Get args, device
    device = torch.device(args.device)

    all_data = []
    for model_name in compvit_models:
        model, config = compvit_factory(model_name=model_name)
        model = model.to(device).eval()
        latency_mean, latency_median, latency_iqr = inference(
            model, device, args.batch_size
        )

        message = f"""\
        ========================
        {colour_text(model_name.upper(), 'green')}
        {colour_text("Parameters", 'cyan')}: {sum(p.numel() for p in model.parameters()):,}
        {colour_text("Depth", 'cyan')}: {config['depth']}
        {colour_text("Embedding Dim", 'cyan')}: {config['embed_dim']}
        {colour_text("num_compressed_tokens", 'cyan')}: {config['num_compressed_tokens']}
        {colour_text("bottleneck_locs", 'cyan')}: {config['bottleneck_locs']}
        {colour_text("bottleneck_size", 'cyan')}: {config['bottleneck_size']}
        {colour_text("bottleneck", 'cyan')}: {config['bottleneck']}
        {colour_text("num_codebook_tokens", 'cyan')}: {config['num_codebook_tokens']}
        {colour_text("inv_bottleneck", 'cyan')}: {config['inv_bottleneck']}
        {colour_text("inv_bottle_size", 'cyan')}: {config['inv_bottle_size']}
        {colour_text("Mean (ms)", "magenta")}: {latency_mean:.2f} 
        {colour_text("Median (ms)", "magenta")}: {latency_median:.2f}
        {colour_text("IQR (ms)", "magenta")}: {latency_iqr:.2f}
        ========================\
        """
        message = textwrap.dedent(message)

        print(message)
        all_data.append(
            {
                "Parameters": sum(p.numel() for p in model.parameters()),
                "Depth": config["depth"],
                "Embedding Dim": config["embed_dim"],
                "num_compressed_tokens": config["num_compressed_tokens"],
                "bottleneck_locs": config["bottleneck_locs"][0],
                "bottleneck_size": config["bottleneck_size"],
                "bottleneck": config["bottleneck"],
                "num_codebook_tokens": config["num_codebook_tokens"],
                "inv_bottleneck": config["inv_bottleneck"],
                "inv_bottle_size": config["inv_bottle_size"],
                "Mean (ms)": latency_mean,
                "Median (ms)": latency_median,
                "IQR (ms)": latency_iqr,
            }
        )
    filename = (
        "_".join(
            [
                device_info(args).replace(" ", ""),
                "compvit",
                f"bs{args.batch_size}",
                f"{args.filetag}",
            ]
        )
        + ".csv"
    )
    export_sweep_data(all_data, filename)


def compvit_sweep(args):
    model_name = args.sweep_model

    ### Get args, device
    device = torch.device(args.device)

    # Create iterators for the two dimensions that we can ablate over.
    # The [None] list is used when we want to fix one dimension.
    token_sweep_iter = [None]
    bottleneck_locs_iter = [None]
    if args.token_sweep:
        token_sweep = args.token_sweep  # argparse will output a list.
        token_sweep_iter = token_sweep  # the iterator is a list.
    if args.bottleneck_locs:
        start, end = args.bottleneck_locs  # argparse will output a 2 element list
        bottleneck_locs_iter = range(start, end + 1)  # the iterator is a range(...)

    all_data = []
    # Use itertools.product this takes the cartesisan product of the two iterators.
    for bottleneck_loc, compressed_tokens in itertools.product(
        bottleneck_locs_iter, token_sweep_iter
    ):
        # Control logic to handle the case when ablating over both dimensions and if one is fixed.
        if bottleneck_loc and compressed_tokens:
            model, config = compvit_factory(
                model_name=model_name,
                num_compressed_tokens=compressed_tokens,
                bottleneck_locs=[bottleneck_loc, bottleneck_loc + 1],
            )
        elif bottleneck_loc:
            model, config = compvit_factory(
                model_name=model_name,
                bottleneck_locs=[bottleneck_loc, bottleneck_loc + 1],
            )
        elif compressed_tokens:
            model, config = compvit_factory(
                model_name=model_name,
                num_compressed_tokens=compressed_tokens,
            )

        # Standard measurement code.
        model = model.to(device).eval()
        latency_mean, latency_median, latency_iqr = inference(
            model, device, args.batch_size
        )

        message = f"""\
        ========================
        {colour_text(model_name.upper(), 'green')}
        {colour_text("Parameters", 'cyan')}: {sum(p.numel() for p in model.parameters()):,}
        {colour_text("Depth", 'cyan')}: {config['depth']}
        {colour_text("Embedding Dim", 'cyan')}: {config['embed_dim']}
        {colour_text("num_compressed_tokens", 'cyan')}: {config['num_compressed_tokens']}
        {colour_text("bottleneck_locs", 'cyan')}: {config['bottleneck_locs']}
        {colour_text("bottleneck_size", 'cyan')}: {config['bottleneck_size']}
        {colour_text("bottleneck", 'cyan')}: {config['bottleneck']}
        {colour_text("num_codebook_tokens", 'cyan')}: {config['num_codebook_tokens']}
        {colour_text("inv_bottleneck", 'cyan')}: {config['inv_bottleneck']}
        {colour_text("inv_bottle_size", 'cyan')}: {config['inv_bottle_size']}
        {colour_text("Mean (ms)", "magenta")}: {latency_mean:.2f}
        {colour_text("Median (ms)", "magenta")}: {latency_median:.2f}
        {colour_text("IQR (ms)", "magenta")}: {latency_iqr:.2f}
        ========================\
        """
        message = textwrap.dedent(message)

        print(message)

        all_data.append(
            {
                "Parameters": sum(p.numel() for p in model.parameters()),
                "Depth": config["depth"],
                "Embedding Dim": config["embed_dim"],
                "num_compressed_tokens": config["num_compressed_tokens"],
                "bottleneck_locs": config["bottleneck_locs"][0],
                "bottleneck_size": config["bottleneck_size"],
                "bottleneck": config["bottleneck"],
                "num_codebook_tokens": config["num_codebook_tokens"],
                "inv_bottleneck": config["inv_bottleneck"],
                "inv_bottle_size": config["inv_bottle_size"],
                "Mean (ms)": latency_mean,
                "Median (ms)": latency_median,
                "IQR (ms)": latency_iqr,
            }
        )

    filename = (
        "_".join(
            [
                device_info(args).replace(" ", ""),
                model_name.replace("_", ""),
                f"bs{args.batch_size}",
                f"{args.filetag}",
            ]
        )
        + ".csv"
    )
    export_sweep_data(all_data, filename)


def test_single(args):
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

    # Inference
    latency_mean, latency_median, latency_iqr = inference(
        model, device, args.batch_size
    )
    print(
        f"{args.model}| Mean/Median/IQR latency (ms) is {latency_mean:.2f} | {latency_median:.2f} | {latency_iqr:.2f}"
    )


def main():
    args = parse_args()
    device_name = device_info(args)
    print(f"{colour_text('Device', 'red')}: {device_name}")

    testing_multiple = args.all_dino or args.all_compvit
    if testing_multiple:
        if args.all_dino:
            print(
                f"{colour_text(f'Benchmarking DINOv2 Models @ batch size = {args.batch_size}.', 'yellow')}"
            )
            test_dino(args)
        if args.all_compvit:
            print(
                f"{colour_text(f'Benchmarking CompViT Models  @ batch size = {args.batch_size}.', 'yellow')}"
            )
            test_compvit(args)
        return 0
    elif args.compvit_sweep:
        message = f"""\
        {colour_text(f'Benchmarking CompViT Models  @ batch size = {args.batch_size}.', 'yellow')}
        {colour_text(f"Sweeping Compressed Tokens {args.token_sweep}.", 'yellow')}
        {colour_text(f"Sweeping Bottleneck Locations from {args.bottleneck_locs}.", "yellow")}\
        """
        print(textwrap.dedent(message))
        compvit_sweep(args)
        return 0
    else:
        test_single(args)
    return 0


if __name__ == "__main__":
    main()
