"""Kernel benchmark of the e2e performance cases.

Compiles each case through the GC GPU pipeline, checks the results against the
NumPy reference, and measures the device time of the kernels with the Level Zero
kernel timestamps over `--iters` runs, after `--warmup` runs. Each case runs in
a separate process, so a crash does not affect the others.

Usage (from python/test, with graph_compiler importable):
    python -m e2e.bench [-k REGEX]... [--warmup N] [--iters N] [--json FILE]

The summary is printed to stderr, `--json` saves the results as a list of:
    {"name", "description", "iters", "warmup", "checked",
     "device_ms": {"min", "median", "mean", "stdev", "max"}, "host_ms": {...},
     "kernels": {kernel: {"launches", "min", ...}}}
or {"name", "description", "error"} for the failed cases.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path


def run_case(case, args):
    from e2e.runner import bench_e2e

    print(f"Running {case.name}: {case.description}", file=sys.stderr, flush=True)
    try:
        inputs = case.inputs()
        result = bench_e2e(
            case.mlir,
            inputs,
            case.reference,
            rtol=case.rtol,
            atol=case.atol,
            outputs=case.outputs(*inputs) if case.outputs else None,
            warmup=args.warmup,
            iters=args.iters,
            check=args.check,
        )
    except Exception as e:
        traceback.print_exc()
        error = str(e).strip().splitlines()
        result = {"error": error[0] if error else type(e).__name__}
    return {"name": case.name, "description": case.description, **result}


def run_isolated(case, args):
    """Run the case in a subprocess, return its result."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        cmd = [
            sys.executable,
            "-m",
            "e2e.bench",
            "--no-isolate",
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--json",
            str(out),
            "-k",
            f"^{case.name}$",
        ]
        if not args.check:
            cmd.append("--no-check")
        env = os.environ.copy()
        root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (root, env.get("PYTHONPATH")) if p
        )
        proc = subprocess.run(cmd, env=env, stdout=sys.stderr)
        if out.exists() and (results := json.loads(out.read_text())):
            return results[0]
        return {
            "name": case.name,
            "description": case.description,
            "error": f"exit code {proc.returncode}",
        }


def summary(results):
    lines = [
        f"{'case':28s} {'median':>9s} {'mean':>9s} {'min':>9s} {'stdev':>8s}"
        f" {'host':>9s}  (device ms per run, host mean ms)"
    ]
    for r in results:
        if "error" in r:
            lines.append(f"{r['name']:28s} FAILED: {r['error']}")
            continue
        d = r["device_ms"]
        lines.append(
            f"{r['name']:28s} {d['median']:9.4f} {d['mean']:9.4f} {d['min']:9.4f}"
            f" {d['stdev']:8.4f} {r['host_ms']['mean']:9.4f}"
        )
        for kernel, k in r["kernels"].items():
            lines.append(
                f"  {kernel:26s} {k['median']:9.4f} {k['mean']:9.4f} {k['min']:9.4f}"
                f" {k['stdev']:8.4f}  x{k['launches']}"
            )
    return "\n".join(lines)


def positive(value):
    if (n := int(value)) <= 0:
        raise argparse.ArgumentTypeError(f"must be positive: {value}")
    return n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "-k",
        dest="patterns",
        action="append",
        default=[],
        metavar="REGEX",
        help="run the cases matching any of the regexes, all by default",
    )
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=positive, default=200)
    ap.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="do not compare the results with the reference",
    )
    ap.add_argument(
        "--no-isolate",
        dest="isolate",
        action="store_false",
        help="run all the cases in this process",
    )
    ap.add_argument("--json", metavar="FILE", help="save the results as JSON")
    ap.add_argument("--list", action="store_true", help="list the cases and exit")
    args = ap.parse_args()
    if args.warmup < 0:
        ap.error("--warmup must not be negative")

    from e2e.perf import select

    cases = select(args.patterns)
    if args.list:
        for case in cases:
            print(f"{case.name:28s} {case.description}")
        return 0
    if not cases:
        ap.error(f"no cases match {args.patterns}")

    run = run_isolated if args.isolate and len(cases) > 1 else run_case
    results = [run(case, args) for case in cases]
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    print(summary(results), file=sys.stderr)
    return 1 if any("error" in r for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
