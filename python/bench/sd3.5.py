#!/usr/bin/env python3
################################################################################
# Copyright (C) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# SPDX-License-Identifier: Apache-2.0
################################################################################
"""SD3.5 transformer latency benchmark with optional per-op GPU profiling.

Usage:
    sd3.5.py [--static] [--profile] [--result out.json]

    # Graph Compiler (MLIR) path
    OV_GPU_ENABLE_MLIR=1 sd3.5.py
"""

import argparse
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openvino as ov

DEFAULT_MODEL = "/models/sd35-medium-fp16-local/transformer/openvino_model.xml"


def static_shape(port, args):
    """Resolve the dynamic dims of an SD3 transformer input:
      hidden_states          [B, 16, H/8, W/8]
      encoder_hidden_states  [B, seq_len, 4096]
      pooled_projections     [B, 2048]
      timestep               [B]
    Static dims are kept, unknown dynamic dims resolve to 1.
    """
    name = port.any_name.lower()
    if "encoder_hidden_states" in name:
        dims = (args.batch, args.seq_len, 4096)
    elif "hidden_states" in name:
        dims = (args.batch, 16, args.height // 8, args.width // 8)
    elif "pooled" in name:
        dims = (args.batch, 2048)
    else:
        dims = (args.batch,)
    return [
        d.get_length() if d.is_static else dims[i] if i < len(dims) else 1
        for i, d in enumerate(port.get_partial_shape())
    ]


def make_request(compiled, inputs):
    req = compiled.create_infer_request()
    for name, tensor in inputs.items():
        req.set_tensor(name, tensor)
    return req


def run(req, iters):
    """Run `iters` inferences, return per-iteration latencies in ms."""
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        # start_async/wait does not copy the outputs back, unlike infer().
        req.start_async()
        req.wait()
        lat.append((time.perf_counter() - t0) * 1e3)
    return lat


def profile(core, model, inputs, args, mean_ms):
    # PERF_COUNT enables event profiling on every primitive, which skews wall-clock
    # latency, hence a separate compiled instance.
    compiled = core.compile_model(model, args.device, {"PERF_COUNT": True})
    req = make_request(compiled, inputs)
    run(req, args.warmup)
    before = {p.node_name: p.real_time.total_seconds() for p in req.profiling_info}
    run(req, args.iters)
    info = req.profiling_info

    # intel_gpu reports a cumulative average over every infer since compile, so
    # back out the warmup share to get the average of the measured window only.
    calls = args.warmup + args.iters
    executed = [p for p in info if p.status == ov.ProfilingInfo.Status.EXECUTED]
    us = {
        p.node_name: (
            p.real_time.total_seconds() * calls
            - before.get(p.node_name, 0.0) * args.warmup
        )
        * 1e6
        / args.iters
        for p in executed
    }
    executed.sort(key=lambda p: us[p.node_name], reverse=True)
    total = sum(us.values()) or 1.0

    status = Counter(p.status.name for p in info)
    print(f"\nNodes: total={len(info)} {dict(status)}")
    # With an in-order queue the per-op intervals are disjoint, so their sum must
    # stay below the wall-clock latency; above it the breakdown is not trustworthy.
    print(
        f"Sum of executed real_time: {total / 1e3:.2f} ms "
        f"({total / 1e3 / mean_ms:.2f} x mean latency)"
    )

    print(f"\nTop {args.top} ops by real_time:")
    print(f"  {'node_type':30s} {'us':>12s} {'%':>6s}  {'exec_type':24s}  node_name")
    for p in executed[: args.top]:
        t = us[p.node_name]
        print(
            f"  {p.node_type:30s} {t:12.1f} {t / total * 100:6.2f}  "
            f"{p.exec_type:24s}  {p.node_name}"
        )

    per_type = defaultdict(lambda: [0.0, 0])
    for p in executed:
        per_type[p.node_type][0] += us[p.node_name]
        per_type[p.node_type][1] += 1
    print("\nBy node_type:")
    print(f"  {'node_type':30s} {'ms':>12s} {'%':>6s} {'count':>7s}")
    for t, (sum_us, n) in sorted(per_type.items(), key=lambda kv: -kv[1][0]):
        print(f"  {t:30s} {sum_us / 1e3:12.2f} {sum_us / total * 100:6.2f} {n:7d}")


def positive(value):
    if (n := int(value)) <= 0:
        raise argparse.ArgumentTypeError(f"must be positive: {value}")
    return n


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--batch", type=positive, default=2, help="CFG doubled batch")
    ap.add_argument("--width", type=positive, default=512)
    ap.add_argument("--height", type=positive, default=512)
    ap.add_argument(
        "--seq-len",
        type=positive,
        default=154,
        help="text tokens (CLIP-L + CLIP-G + T5 joint)",
    )
    ap.add_argument("--warmup", type=positive, default=2)
    ap.add_argument("--iters", type=positive, default=5)
    ap.add_argument(
        "--static",
        action="store_true",
        help="reshape a dynamic model to static shapes before compiling",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="print per-op timings from a PERF_COUNT compiled instance",
    )
    ap.add_argument(
        "--top", type=positive, default=40, help="number of ops to print with --profile"
    )
    ap.add_argument("--result", metavar="FILE", help="write latency stats as JSON")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"OV version: {ov.get_version()}")
    print(f"Device:     {args.device}")
    print(f"Model:      {args.model}")
    for k in sorted(k for k in os.environ if k.startswith(("OV_GPU_", "OV_MLIR"))):
        print(f"{k}={os.environ[k]}")

    core = ov.Core()
    model = core.read_model(args.model)
    print(f"Source model is {'dynamic' if model.is_dynamic() else 'static'}")

    if args.static:
        shapes = {p: ov.PartialShape(static_shape(p, args)) for p in model.inputs}
        print("Reshaping to static shapes:")
        for p, shape in shapes.items():
            print(f"  {p.any_name:40s} {p.get_partial_shape()} -> {shape}")
        model.reshape(shapes)
        if model.is_dynamic():
            print("WARNING: model is still dynamic after reshape")

    print("Compiling model...")
    compiled = core.compile_model(model, args.device)

    rng = np.random.default_rng(0)
    inputs = {
        p.any_name: ov.Tensor(
            (rng.standard_normal(static_shape(p, args)) * 0.1).astype(
                p.get_element_type().to_dtype()
            )
        )
        for p in compiled.inputs
    }
    print("\nInputs:")
    for name, t in inputs.items():
        print(f"  {name:40s} {t.get_element_type().get_type_name()} {list(t.shape)}")

    req = make_request(compiled, inputs)
    run(req, args.warmup)
    lat = run(req, args.iters)
    stats = {
        "min_ms": min(lat),
        "median_ms": statistics.median(lat),
        "mean_ms": statistics.fmean(lat),
        "stdev_ms": statistics.stdev(lat) if len(lat) > 1 else 0.0,
        "max_ms": max(lat),
        "iters": len(lat),
    }
    print(
        "\nLatency (ms): "
        + " ".join(f"{k[:-3]}={v:.3f}" for k, v in stats.items() if k.endswith("_ms"))
        + f" n={len(lat)}"
    )

    if args.result:
        Path(args.result).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "device": args.device,
                    "ov_version": ov.get_version(),
                    **stats,
                },
                indent=2,
            )
        )

    if args.profile:
        profile(core, model, inputs, args, stats["mean_ms"])


if __name__ == "__main__":
    main()
