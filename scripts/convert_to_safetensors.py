from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import tempfile
import urllib.request

import torch
from safetensors.torch import save_file

DEFAULT_MODEL_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if hasattr(checkpoint, "module"):
        return checkpoint.module.state_dict()

    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()

    if isinstance(checkpoint, Mapping):
        if "model" in checkpoint:
            return extract_state_dict(checkpoint["model"])
        if "state_dict" in checkpoint:
            return extract_state_dict(checkpoint["state_dict"])
        if all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in checkpoint.items()):
            return dict(checkpoint)

    raise TypeError("Unsupported checkpoint format; could not extract a state dict.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PyTorch checkpoint to safetensors format.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/rmvpe.safetensors"),
        help="Path to write the converted safetensors file.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "rmvpe.pt"
        urllib.request.urlretrieve(DEFAULT_MODEL_URL, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)
        tensors = {key: value.detach().cpu().contiguous() for key, value in state_dict.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"source_checkpoint": DEFAULT_MODEL_URL}
    save_file(tensors, str(args.output), metadata=metadata)

    print(f"Saved {len(tensors)} tensors to {args.output}")


if __name__ == "__main__":
    main()
