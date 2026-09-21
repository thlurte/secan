#!/usr/bin/env python3
"""
Memory Mountain Sweep Automation & Cache Hierarchy Profiler.
Sweeps working set sizes from 4 KB to 64 MB across strides 1 to 64 elements,
recording effective memory throughput across L1, L2, L3, and DRAM.
"""

import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def get_hardware_info():
    uname = platform.uname()
    return {
        "platform": f"{uname.system} {uname.release} ({uname.machine})",
        "processor": uname.processor or "AMD Zen 4 (Hawk Point)",
        "l1d_cache": "32 KiB (x6)",
        "l1i_cache": "32 KiB (x6)",
        "l2_cache": "1024 KiB (x6)",
        "l3_cache": "16384 KiB (x1)",
    }


def run_memory_mountain_sweep(secan_root: Path, output_file: Path):
    binary_path = secan_root / "build" / "benchmarks" / "bench_exact_scan"
    if not binary_path.exists():
        print(f"Error: Benchmark binary not found at {binary_path}. Build it first.")
        sys.exit(1)

    sizes_kb = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536]
    strides = [1, 2, 4, 8, 16, 32, 64]
    benchmarks_data = []

    for sz in sizes_kb:
        for st in strides:
            filter_pattern = f"^BM_MemoryMountain/{sz}/{st}$"
            cmd = [str(binary_path), f"--benchmark_filter={filter_pattern}", "--benchmark_format=json"]
            print(f"Running Memory Mountain: size={sz} KB, stride={st}...")
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            data = json.loads(res.stdout)
            for b in data.get("benchmarks", []):
                bytes_per_sec = b.get("bytes_per_second", 0.0)
                gib_per_sec = bytes_per_sec / (1024.0 ** 3)
                benchmarks_data.append({
                    "name": b.get("name"),
                    "size_kb": sz,
                    "stride": st,
                    "time_ns": round(b.get("real_time", 0.0), 2),
                    "cpu_ns": round(b.get("cpu_time", 0.0), 2),
                    "iterations": b.get("iterations", 0),
                    "bytes_per_second_gib": round(gib_per_sec, 4),
                })

    payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware": get_hardware_info(),
        "benchmarks": benchmarks_data,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Successfully recorded memory mountain sweep to {output_file}")


def main():
    secan_root = Path("/home/ahmed/personal/secan")
    output_file = Path("/home/ahmed/personal/goals/research/2026-09-measurement-protocol/notes/week3_memory_mountain.json")
    run_memory_mountain_sweep(secan_root, output_file)


if __name__ == "__main__":
    main()
