import argparse
import os
import sys

import torch


def _load_model_factory(model_name: str):
    from src.model_lib.MiniFASNet import (
        MiniFASNetV1,
        MiniFASNetV2,
        MiniFASNetV1SE,
        MiniFASNetV2SE,
    )

    mapping = {
        "MiniFASNetV1": MiniFASNetV1,
        "MiniFASNetV2": MiniFASNetV2,
        "MiniFASNetV1SE": MiniFASNetV1SE,
        "MiniFASNetV2SE": MiniFASNetV2SE,
    }
    if model_name not in mapping:
        raise ValueError(f"Unsupported model name: {model_name}")
    return mapping[model_name]


def _load_weights(model: torch.nn.Module, model_path: str) -> None:
    state_dict = torch.load(model_path, map_location="cpu")

    # Some checkpoints are saved with DataParallel prefix "module.".
    if state_dict:
        first_key = next(iter(state_dict.keys()))
        if first_key.startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MiniFASNet .pth to ONNX")
    parser.add_argument("--repo", required=True, help="Path to Silent-Face-Anti-Spoofing repo root")
    parser.add_argument("--weights", required=True, help="Path to .pth weights file")
    parser.add_argument("--model-name", required=True, help="MiniFASNetV1 | MiniFASNetV2 | MiniFASNetV1SE | MiniFASNetV2SE")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo)
    weights_path = os.path.abspath(args.weights)
    output_path = os.path.abspath(args.output)

    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"Repo path does not exist: {repo_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

    # Import official MiniFASNet definitions from the downloaded repo.
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from src.utility import get_kernel, parse_model_name

    checkpoint_name = os.path.basename(weights_path)
    h_input, w_input, ckpt_model_type, _ = parse_model_name(checkpoint_name)
    conv6_kernel = get_kernel(h_input, w_input)

    if ckpt_model_type != args.model_name:
        raise ValueError(
            f"Checkpoint model type '{ckpt_model_type}' does not match --model-name '{args.model_name}'"
        )

    model_factory = _load_model_factory(args.model_name)
    # Keep model defaults (including class count) and only set the conv6 kernel.
    model = model_factory(conv6_kernel=conv6_kernel)
    _load_weights(model, weights_path)
    model.eval()

    dummy = torch.randn(1, 3, args.height, args.width)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    print(f"Export complete: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
