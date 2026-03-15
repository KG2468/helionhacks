import argparse
import dataclasses
import inspect
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import helion
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import submission


def _shape_key(shape_text: str) -> tuple[int, int, int, int, int]:
    parts = [p.strip() for p in shape_text.split(",")]
    if len(parts) != 5:
        raise ValueError("Shape must be B,T,H,K,V")
    return tuple(int(p) for p in parts)


def _make_inputs(
    shape: tuple[int, int, int, int, int],
    chunk_size: int,
    device: str,
    dtype: torch.dtype,
):
    b, t, h, k, v = shape
    scale = 0.05
    q_t = torch.randn(b, t, h, k, device=device, dtype=dtype) * scale
    k_t = torch.randn(b, t, h, k, device=device, dtype=dtype) * scale
    v_t = torch.randn(b, t, h, v, device=device, dtype=dtype) * scale
    nt = (t + chunk_size - 1) // chunk_size
    h_t = torch.randn(b, nt, h, k, v, device=device, dtype=dtype) * scale
    g_t = -2.0 + 0.1 * torch.randn(b, t, h, device=device, dtype=dtype)
    inv_scale = k ** -0.5
    return q_t, k_t, v_t, h_t, g_t, inv_scale


def _extract_possible_configs(obj) -> list[str]:
    out: list[str] = []
    if obj is None:
        return out

    for name in ["best_config", "config", "best"]:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                out.append(str(value))

    if isinstance(obj, dict):
        for key in ["best_config", "config", "best"]:
            if key in obj and obj[key] is not None:
                out.append(str(obj[key]))

    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, dict)):
        for item in obj:
            if isinstance(item, tuple) and len(item) >= 2:
                out.append(str(item[1]))

    return list(dict.fromkeys(out))


def _autotune_kernel(kernel, args: tuple[torch.Tensor, ...]):
    calls = [
        (args, {}),
        ((args,), {}),
        ((), {"args": args}),
    ]

    errors: list[str] = []
    for positional, kwargs in calls:
        try:
            return kernel.autotune(*positional, **kwargs)
        except TypeError as exc:
            errors.append(str(exc))

    signature = "<unknown>"
    try:
        signature = str(inspect.signature(kernel.autotune))
    except Exception:
        pass

    detail = "\n".join(errors) if errors else "No compatible call pattern matched"
    raise RuntimeError(f"Unable to call kernel.autotune with signature {signature}.\n{detail}")


def _resolve_acfs(args: argparse.Namespace) -> list[str | None]:
    if not args.compileiq:
        return [None]

    if args.acf:
        acfs: list[str] = []
        for item in args.acf:
            p = Path(item)
            if not p.is_absolute():
                p = Path(args.acf_dir) / p
            acfs.append(str(p))
        return acfs

    acf_dir = Path(args.acf_dir)
    discovered = sorted(str(path) for path in acf_dir.glob("*.acf"))
    if args.max_acfs > 0:
        discovered = discovered[: args.max_acfs]

    if discovered:
        return discovered

    print(
        f"CompileIQ requested, but no .acf files found under {acf_dir}. "
        "Falling back to config without advanced_controls_file."
    )
    return [None]


def _config_to_kwargs(config: helion.Config) -> dict:
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)

    for method_name in ["model_dump", "dict", "_asdict"]:
        method = getattr(config, method_name, None)
        if callable(method):
            return dict(method())

    return {k: v for k, v in vars(config).items() if not k.startswith("_")}


def _config_with_acf(base_config: helion.Config, acf: str | None) -> helion.Config:
    if acf is None:
        return base_config

    kwargs = _config_to_kwargs(base_config)
    kwargs["advanced_controls_file"] = acf
    return type(base_config)(**kwargs)


def _build_kernel(shape: tuple[int, int, int, int, int], acf: str | None):
    _, t, _, k, v = shape
    config = submission.SHAPE_CONFIGS[shape]
    config = _config_with_acf(config, acf)
    chunk_size = submission._pick_chunk_size(k, v, t)
    return submission._make_kernel(config, chunk_size), chunk_size


def _apply_runtime_env(args: argparse.Namespace):
    os.environ["HELION_AUTOTUNE_EFFORT"] = args.effort

    if args.tileir:
        os.environ["ENABLE_TILE"] = "1"
        os.environ["HELION_BACKEND"] = "tileir"


def main():
    parser = argparse.ArgumentParser(description="Autotune kernels declared in submission.py")
    parser.add_argument(
        "--shape",
        action="append",
        help="Shape in B,T,H,K,V format. Repeat to run multiple shapes. Default: all shapes in SHAPE_CONFIGS",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run only larger benchmark-like shapes (T >= 512)",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument(
        "--effort",
        default="full",
        choices=["none", "quick", "full"],
        help="Sets HELION_AUTOTUNE_EFFORT",
    )
    parser.add_argument(
        "--tileir",
        action="store_true",
        help="Enable TileIR backend via ENABLE_TILE=1 and HELION_BACKEND=tileir",
    )
    parser.add_argument(
        "--compileiq",
        action="store_true",
        help="Try CompileIQ ACFs by setting advanced_controls_file in config",
    )
    parser.add_argument(
        "--acf",
        action="append",
        help="ACF file path (absolute or relative to --acf-dir). Repeat to try multiple ACFs.",
    )
    parser.add_argument(
        "--acf-dir",
        default="/opt/booster_pack",
        help="Directory used to resolve relative --acf paths and auto-discover ACFs",
    )
    parser.add_argument(
        "--max-acfs",
        type=int,
        default=8,
        help="Maximum number of auto-discovered ACF files when --compileiq is used without --acf",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    _apply_runtime_env(args)

    dtype = getattr(torch, args.dtype)
    if args.shape:
        shapes = [_shape_key(s) for s in args.shape]
    else:
        shapes = list(submission.SHAPE_CONFIGS.keys())

    if args.benchmark_only:
        shapes = [shape for shape in shapes if shape[1] >= 512]

    if not shapes:
        raise RuntimeError("No shapes selected")

    acfs = _resolve_acfs(args)

    print(f"Selected shapes ({len(shapes)}):")
    for shape in shapes:
        print(f"  - {shape}")
    print(f"Autotune effort: {args.effort}")
    if args.tileir:
        print("TileIR: enabled (ENABLE_TILE=1, HELION_BACKEND=tileir)")
    if args.compileiq:
        print("CompileIQ ACF candidates:")
        for acf in acfs:
            print(f"  - {acf if acf else '<none>'}")

    for shape in shapes:
        for acf in acfs:
            acf_label = acf if acf else "<none>"
            print(f"\n=== Autotuning shape {shape} (acf={acf_label}) ===")
            kernel, chunk_size = _build_kernel(shape, acf)
            kernel_args = _make_inputs(shape, chunk_size=chunk_size, device=args.device, dtype=dtype)
            result = _autotune_kernel(kernel, kernel_args)

            configs = _extract_possible_configs(result)
            if configs:
                print("Best/returned config candidates:")
                for cfg in configs:
                    print(f"  {cfg}")
            else:
                print("Autotune completed; no config object found in return value.")


if __name__ == "__main__":
    main()
