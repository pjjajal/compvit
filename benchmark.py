import torch
import argparse

from torch.profiler import profile, record_function, ProfilerActivity
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
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"])

    return parser.parse_args()


def main(args):
    if "dinov2" in args.model:
        model, _ = dinov2_factory(args.model)
    elif "compvit" in args.model:
        model, _ = compvit_factory(args.model)

    model = model.to(device=args.device)
    model = model.eval()
    inputs = torch.randn((args.batchsize, 3, 224, 224)).to(device=args.device)

    print(f"# of parameters: {sum(p.numel() for p in model.parameters()):_}")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        # with_stack=True,
        # with_modules=True
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                model(inputs)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # prof.export_chrome_trace(f"{args.model}_v2_10.json")

if __name__ == "__main__":
    args = parse_args()
    main(args)
