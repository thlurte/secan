#!/usr/bin/env python3
import argparse
import struct
import numpy as np

def write_fvecs(filename, data):
    with open(filename, "wb") as f:
        dim = data.shape[1]
        for vec in data:
            f.write(struct.pack("i", dim))
            f.write(vec.astype(np.float32).tobytes())

def write_ivecs(filename, data):
    with open(filename, "wb") as f:
        dim = data.shape[1]
        for vec in data:
            f.write(struct.pack("i", dim))
            f.write(vec.astype(np.int32).tobytes())

def write_bvecs(filename, data):
    with open(filename, "wb") as f:
        dim = data.shape[1]
        for vec in data:
            f.write(struct.pack("i", dim))
            f.write(vec.astype(np.uint8).tobytes())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic .fvecs/.bvecs/.ivecs datasets")
    parser.add_argument("--format", choices=["fvecs", "bvecs", "ivecs"], default="fvecs")
    parser.add_argument("--num", type=int, default=10000, help="Number of vectors")
    parser.add_argument("--dim", type=int, default=128, help="Dimensionality")
    parser.add_argument("--output", type=str, required=True, help="Output filepath")
    args = parser.parse_args()

    np.random.seed(42)
    if args.format == "fvecs":
        data = np.random.randn(args.num, args.dim).astype(np.float32)
        write_fvecs(args.output, data)
    elif args.format == "bvecs":
        data = np.random.randint(0, 256, size=(args.num, args.dim), dtype=np.uint8)
        write_bvecs(args.output, data)
    elif args.format == "ivecs":
        data = np.random.randint(0, 1000000, size=(args.num, args.dim), dtype=np.int32)
        write_ivecs(args.output, data)
    print(f"Generated {args.num} x {args.dim} {args.format} at {args.output}")
