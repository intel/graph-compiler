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
"""Merge the benchmark results into a report, print it as markdown tables and
compare it with a baseline report.

The report is a JSON object {"version": 1, "ov": [...], "kernels": [...]}:
- "ov": the OpenVINO model benchmarks, `python/bench/sd3.5.py --result`;
- "kernels": the GC kernel benchmarks, `python -m e2e.bench --json`.

Usage:
    bench-report.py [--ov FILE]... [--kernels FILE]... [--input REPORT]...
                    [--baseline REPORT] [--save REPORT] [--embed] [--annotate]
    bench-report.py --extract MARKDOWN --save REPORT

With --embed, the report is appended to the markdown as an HTML comment, that
--extract finds in a markdown file, for example, a pull request body, to be used
as a baseline later.
"""

import argparse
import json
import re
import sys
from pathlib import Path

VERSION = 1
EMBED_RE = re.compile(r"<!-- bench-report: (.*?) -->", re.DOTALL)


def compact_ov(r):
    keys = ("name", "static", "mean_ms", "median_ms", "min_ms", "stdev_ms")
    keys += ("mlir_ms", "mlir_count", "profile_sum_ms")
    return {k: r[k] for k in keys if k in r}


def compact_kernel(r):
    out = {k: r[k] for k in ("name", "description", "error") if k in r}
    if "error" in r:
        return out
    out["iters"] = r["iters"]
    out["device_ms"] = {
        k: r["device_ms"][k] for k in ("median", "mean", "min", "stdev", "max")
    }
    out["host_ms"] = {k: r["host_ms"][k] for k in ("median", "mean")}
    out["kernels"] = {
        name: {k: v[k] for k in ("launches", "median", "mean", "min", "stdev")}
        for name, v in r["kernels"].items()
    }
    return out


def load(path):
    return json.loads(Path(path).read_text())


def merge(args):
    report = {"version": VERSION, "ov": [], "kernels": []}
    for path in args.input:
        data = load(path)
        report["ov"] += data.get("ov", [])
        report["kernels"] += data.get("kernels", [])
    for path in args.ov:
        data = load(path)
        report["ov"] += [
            compact_ov(r) for r in (data if isinstance(data, list) else [data])
        ]
    for path in args.kernels:
        report["kernels"] += [compact_kernel(r) for r in load(path)]
    return report


def by_name(entries):
    return {e["name"]: e for e in entries if e.get("name")}


def delta(cur, base):
    if cur is None or not base:
        return None
    return (cur - base) / base * 100


class Table:
    def __init__(self, threshold):
        self.threshold = threshold
        self.regressions = []
        self.improvements = []

    def cell(self, label, cur, base):
        """Format the relative change, a positive one is slower."""
        d = delta(cur, base)
        if d is None:
            return "new" if cur is not None and base is None else ""
        text = f"{d:+.1f}%"
        if d > self.threshold:
            self.regressions.append(f"{label} {text}")
            return f"**{text}**"
        if d < -self.threshold:
            self.improvements.append(f"{label} {text}")
            return f"_{text}_"
        return text


def ms(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def header(*cols):
    return [
        "| " + " | ".join(cols) + " |",
        "|---|" + "--:|" * (len(cols) - 1),
    ]


def render_ov(report, baseline, t):
    if not report["ov"]:
        return []
    base = by_name(baseline["ov"]) if baseline else None
    cmp = base is not None
    lines = ["#### OpenVINO (ms per inference)", ""]
    lines += header(
        "Benchmark",
        "Mean",
        *(["Δ mean"] if cmp else []),
        "MLIROp",
        "Count",
        *(["Δ MLIROp"] if cmp else []),
    )
    for r in report["ov"]:
        name = r.get("name") or "?"
        b = (base or {}).get(name) or {}
        mean, mlir = r.get("mean_ms"), r.get("mlir_ms")
        cells = [name, ms(mean)]
        if cmp:
            cells.append(t.cell(name, mean, b.get("mean_ms")))
        cells += [ms(mlir), str(r.get("mlir_count") or "-")]
        if cmp:
            cells.append(
                t.cell(f"{name} MLIROp", mlir, b.get("mlir_ms")) if mlir else ""
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "MLIROp: the total time of the MLIROp nodes, measured with PERF_COUNT.",
        "",
    ]
    return lines


def render_kernels(report, baseline, t):
    if not report["kernels"]:
        return []
    base = by_name(baseline["kernels"]) if baseline else None
    cmp = base is not None
    lines = ["#### GC kernels (device ms per run, L0 kernel timestamps)", ""]
    lines += header(
        "Case",
        "Median",
        *(["Δ median"] if cmp else []),
        "Mean",
        "Min",
        "Stdev",
        "Host mean",
    )

    def row(label, key, cur, b, host):
        cells = [label, ms(cur["median"], 4)]
        if cmp:
            cells.append(t.cell(key, cur["median"], b and b["median"]))
        cells += [ms(cur[k], 4) for k in ("mean", "min", "stdev")]
        cells.append(ms(host, 4) if host is not None else "")
        return "| " + " | ".join(cells) + " |"

    for r in report["kernels"]:
        name = r["name"]
        if "error" in r:
            lines.append(f"| {name} | FAILED: {r['error']} |")
            t.regressions.append(f"{name} FAILED")
            continue
        b = (base or {}).get(name) or {}
        host = r["host_ms"]["mean"]
        lines.append(row(name, name, r["device_ms"], b.get("device_ms"), host))
        if len(r["kernels"]) > 1:
            for kernel, k in r["kernels"].items():
                label = f"&nbsp;&nbsp;`{kernel}` x{k['launches']}"
                kb = b.get("kernels", {}).get(kernel)
                lines.append(row(label, f"{name}/{kernel}", k, kb, None))
    lines.append("")
    return lines


def render(report, baseline, label, threshold):
    t = Table(threshold)
    lines = []
    if baseline is not None:
        lines += [
            f"Δ: the change relative to {label or 'the baseline'}, "
            f"in bold if slower by more than {threshold:g}%.",
            "",
        ]
    lines += render_ov(report, baseline, t)
    lines += render_kernels(report, baseline, t)
    if baseline is not None:
        lines.append(
            f"**Regressions:** {', '.join(t.regressions)}"
            if t.regressions
            else "No regressions."
        )
        if t.improvements:
            lines += ["", f"**Improvements:** {', '.join(t.improvements)}"]
        lines.append("")
    return "\n".join(lines), t


def embed(report):
    # "--" would end the HTML comment, it can only occur in the JSON strings.
    text = json.dumps(report, separators=(",", ":")).replace("--", "-\\u002d")
    return f"<!-- bench-report: {text} -->"


def extract(text):
    # The first one, the text may quote the older reports below.
    match = EMBED_RE.search(text)
    return json.loads(match.group(1)) if match else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ov", action="append", default=[], metavar="FILE")
    ap.add_argument("--kernels", action="append", default=[], metavar="FILE")
    ap.add_argument("--input", action="append", default=[], metavar="REPORT")
    ap.add_argument("--baseline", metavar="REPORT")
    ap.add_argument(
        "--baseline-label", metavar="TEXT", help="e.g. a link to the baseline"
    )
    ap.add_argument("--threshold", type=float, default=5.0, metavar="PCT")
    ap.add_argument("--save", metavar="REPORT", help="save the merged report")
    ap.add_argument(
        "--embed", action="store_true", help="embed the report in the output"
    )
    ap.add_argument(
        "--annotate",
        action="store_true",
        help="print a GitHub Actions warning for each regression to stderr",
    )
    ap.add_argument(
        "--extract",
        metavar="MARKDOWN",
        help="extract an embedded report, '-' for stdin, to --save; exit 1 if not found",
    )
    args = ap.parse_args()

    if args.extract:
        if not args.save:
            ap.error("--extract requires --save")
        src = sys.stdin if args.extract == "-" else open(args.extract)
        if (report := extract(src.read())) is None:
            print("No embedded benchmark report found", file=sys.stderr)
            return 1
        Path(args.save).write_text(json.dumps(report, indent=2))
        return 0

    report = merge(args)
    baseline = None
    if args.baseline:
        baseline = load(args.baseline)
        if baseline.get("version") != VERSION:
            print(
                f"Ignoring baseline version {baseline.get('version')}", file=sys.stderr
            )
            baseline = None
    if args.save:
        Path(args.save).write_text(json.dumps(report, indent=2))

    text, t = render(report, baseline, args.baseline_label, args.threshold)
    print(text)
    if args.embed:
        print(embed(report))
    if args.annotate:
        for r in t.regressions:
            print(f"::warning title=Performance regression::{r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
