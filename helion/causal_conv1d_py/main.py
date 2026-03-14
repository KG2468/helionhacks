import argparse
import inspect
from collections.abc import Iterable

import torch

import submission


def _shape_key(shape_text: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in shape_text.split(",")]
    if len(parts) != 4:
        raise ValueError("Shape must be B,D,S,W")
    b, d, s, w = (int(p) for p in parts)
    return b, d, s, w


def _make_inputs(shape: tuple[int, int, int, int], device: str, dtype: torch.dtype):
    b, d, s, w = shape
    x = torch.randn(b, d, s, device=device, dtype=dtype)
    weight = torch.randn(d, w, device=device, dtype=dtype)
    bias = torch.randn(d, device=device, dtype=dtype)
    pad = torch.zeros(b, d, w - 1, device=device, dtype=dtype)
    x_pad = torch.cat([pad, x], dim=2)
    return x_pad, weight, bias


def _build_kernel(shape: tuple[int, int, int, int]):
    config = submission.SHAPE_CONFIGS[shape]
    return submission._make_kernel(config)


def _extract_possible_configs(obj) -> list[str]:
    out: list[str] = []
    if obj is None:
        return out

    direct = [
        "best_config",
        "config",
        "best",
    ]
    for name in direct:
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

    failures: list[str] = []
    for positional, kwargs in calls:
        try:
            return kernel.autotune(*positional, **kwargs)
        except TypeError as exc:
            failures.append(str(exc))

    signature = "<unknown>"
    try:
        signature = str(inspect.signature(kernel.autotune))
    except Exception:
        pass

    msg = "\n".join(failures) if failures else "No compatible call pattern matched"
    raise RuntimeError(f"Unable to call kernel.autotune with signature {signature}.\n{msg}")


def main():
    parser = argparse.ArgumentParser(description="Autotune kernels declared in submission.py")
    parser.add_argument(
        "--shape",
        action="append",
        help="Shape in B,D,S,W format. Repeat to run multiple shapes. Default: all shapes in SHAPE_CONFIGS",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run only benchmark shapes (D >= 512)",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    dtype = getattr(torch, args.dtype)
    if args.shape:
        shapes = [_shape_key(s) for s in args.shape]
    else:
        shapes = list(submission.SHAPE_CONFIGS.keys())

    if args.benchmark_only:
        shapes = [shape for shape in shapes if shape[1] >= 512]

    if not shapes:
        raise RuntimeError("No shapes selected")

    print(f"Selected shapes ({len(shapes)}):")
    for shape in shapes:
        print(f"  - {shape}")

    for shape in shapes:
        print(f"\n=== Autotuning shape {shape} ===")
        kernel = _build_kernel(shape)
        kernel_args = _make_inputs(shape, device=args.device, dtype=dtype)
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
