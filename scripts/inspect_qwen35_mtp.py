#!/usr/bin/env python3
"""
Inspect MTP tensor names and shapes in a Qwen3.5 HuggingFace model.

Run this on a machine with access to the HuggingFace model:
    python3 scripts/inspect_qwen35_mtp.py /path/to/Qwen3.5-0.8B

Or download first:
    huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir /tmp/Qwen3.5-0.8B
    python3 scripts/inspect_qwen35_mtp.py /tmp/Qwen3.5-0.8B
"""

import sys
import json
from pathlib import Path


def inspect_mtp_tensors(model_path: str):
    model_dir = Path(model_path)

    # Load config
    config_file = model_dir / "config.json"
    if not config_file.exists():
        config_file = model_dir / "text_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print("=== Config ===")
        for key in sorted(config):
            if any(x in key.lower() for x in ["mtp", "nextn", "layer", "hidden", "embed", "head", "full_attention"]):
                print(f"  {key}: {config[key]}")
        print()
    else:
        print(f"WARNING: No config.json found at {model_dir}")

    # Find safetensors files
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        # Try model subdirectory
        safetensor_files = sorted(model_dir.glob("*/*.safetensors"))
    if not safetensor_files:
        print("ERROR: No .safetensors files found")
        sys.exit(1)

    print(f"=== Safetensors files ===")
    for f in safetensor_files:
        print(f"  {f.name}")
    print()

    # Read tensor index if available
    index_file = model_dir / "model.safetensors.index.json"
    tensor_to_file: dict[str, str] = {}
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        tensor_to_file = index.get("weight_map", {})

    # Import safetensors
    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: safetensors not installed. Run: pip install safetensors")
        sys.exit(1)

    # Collect all tensor names and shapes
    all_tensors: dict[str, tuple[list[int], str]] = {}  # name -> (shape, dtype)
    for sf_file in safetensor_files:
        with safe_open(str(sf_file), framework="numpy") as f:
            for name in f.keys():
                tensor = f.get_slice(name)
                all_tensors[name] = (list(tensor.get_shape()), str(tensor.get_dtype()))

    # Separate MTP from main model tensors
    mtp_tensors = {k: v for k, v in all_tensors.items() if k.startswith("mtp")}
    main_tensors = {k: v for k, v in all_tensors.items() if not k.startswith("mtp")}

    print(f"=== Summary ===")
    print(f"  Total tensors:      {len(all_tensors)}")
    print(f"  Main model tensors: {len(main_tensors)}")
    print(f"  MTP tensors:        {len(mtp_tensors)}")
    print()

    if not mtp_tensors:
        print("WARNING: No MTP tensors found (all tensor names listed below)")
        print("\nAll tensor names (first 20):")
        for name in sorted(all_tensors)[:20]:
            shape, dtype = all_tensors[name]
            print(f"  {name:<60} {shape} {dtype}")
        return

    print("=== MTP Tensor Names and Shapes ===")
    for name in sorted(mtp_tensors):
        shape, dtype = mtp_tensors[name]
        print(f"  {name:<70} {shape}")

    # Analysis: group by MTP head index
    print("\n=== MTP Structure Analysis ===")
    import re
    head_pattern = re.compile(r"^mtp\.(\d+)\.")
    shared_pattern = re.compile(r"^mtp\.([a-zA-Z_]+)")

    mtp_by_head: dict[int, list[str]] = {}
    mtp_shared: list[str] = []

    for name in sorted(mtp_tensors):
        head_match = head_pattern.match(name)
        if head_match:
            head_idx = int(head_match.group(1))
            if head_idx not in mtp_by_head:
                mtp_by_head[head_idx] = []
            mtp_by_head[head_idx].append(name)
        else:
            mtp_shared.append(name)

    if mtp_by_head:
        print(f"  MTP heads: {sorted(mtp_by_head.keys())}")
        for head_idx, names in sorted(mtp_by_head.items()):
            print(f"\n  Head {head_idx}:")
            for name in names:
                shape, dtype = mtp_tensors[name]
                suffix = name[len(f"mtp.{head_idx}."):]
                print(f"    .{suffix:<60} {shape}")
    else:
        print("  No indexed MTP heads found. Shared tensors:")
        for name in mtp_shared:
            shape, dtype = mtp_tensors[name]
            print(f"    {name:<60} {shape}")

    # Also print a few main model layer tensors for comparison
    print("\n=== Main Model Layer Structure (first layer, for comparison) ===")
    layer0 = {k: v for k, v in main_tensors.items() if "layers.0." in k or "model.layers.0." in k}
    for name in sorted(layer0)[:15]:
        shape, dtype = layer0[name]
        print(f"  {name:<70} {shape}")

    # Config suggestions for converter
    num_hidden_layers = config.get("num_hidden_layers") if config_file.exists() else "?"
    num_mtp_heads = len(mtp_by_head) if mtp_by_head else len(mtp_shared)
    print(f"\n=== Converter Config Suggestions ===")
    print(f"  num_hidden_layers: {num_hidden_layers}")
    print(f"  num MTP heads:     {num_mtp_heads}")
    print(f"  => block_count = num_hidden_layers + num_mtp_heads = {num_hidden_layers} + {num_mtp_heads}")
    print()
    print("Copy this output to help implement the MTP converter.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    inspect_mtp_tensors(sys.argv[1])
