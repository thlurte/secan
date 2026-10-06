import numpy as np
import struct

def load_fvecs(filename: str, max_vectors: int = 100000) -> np.ndarray:
    """
    Reads a standard .fvecs binary file into a float32 numpy array of shape (N, D).
    Binary layout: [int32: dim][float32 * dim][int32: dim][float32 * dim]...
    """
    vectors = []
    with open(filename, 'rb') as f:
        while len(vectors) < max_vectors:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack(f'{dim}f', f.read(dim * 4))
            vectors.append(vec)
    return np.array(vectors, dtype=np.float32)

def analyze_embedding_geometry(matrix: np.ndarray, name: str) -> dict:
    """
    Computes geometric properties of a vector dataset:
    1. Mean & Std pairwise cosine similarity (Cone width)
    2. SVD singular value variance decay (Top-10 variance ratio)
    """
    num_vectors, dim = matrix.shape
    # 1. Normalize all vectors to unit length (L2 norm = 1.0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed = matrix / np.maximum(norms, 1e-12)
    # 2. Sample 10,000 random pairs to compute cosine distribution
    num_samples = min(10000, num_vectors)
    idx1 = np.random.choice(num_vectors, num_samples, replace=False)
    idx2 = np.random.choice(num_vectors, num_samples, replace=False)
    cosines = np.sum(normed[idx1] * normed[idx2], axis=1)
    # 3. Center the matrix and run Singular Value Decomposition (SVD)
    centered = normed - np.mean(normed, axis=0)
    # Use full_matrices=False for speed
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    
    # Variance explained by each singular value is s_i^2
    total_variance = np.sum(s ** 2)
    top10_variance = np.sum(s[:min(10, len(s))] ** 2)
    top10_ratio = float(top10_variance / max(total_variance, 1e-12))
    return {
        "dataset": name,
        "num_vectors": num_vectors,
        "dimension": dim,
        "mean_pairwise_cosine": float(np.mean(cosines)),
        "std_pairwise_cosine": float(np.std(cosines)),
        "top10_singular_variance_ratio": top10_ratio,
        "is_anisotropic": bool(np.mean(cosines) > 0.25 or top10_ratio > 0.50)
    }


def generate_synthetic_data(n: int = 10000, d: int = 128, cone_strength: float = 0.0) -> np.ndarray:
    """
    If cone_strength = 0.0: Generates uniform isotropic Gaussian vectors (Sphere).
    If cone_strength > 0.0: Pulls all vectors along a dominant direction (Cone).
    """
    # 1. Start with random isotropic Gaussian noise
    data = np.random.randn(n, d).astype(np.float32)
    
    if cone_strength > 0.0:
        # Create a dominant shared direction vector
        dominant_direction = np.ones((1, d), dtype=np.float32)
        dominant_direction /= np.linalg.norm(dominant_direction)
        # Shift all vectors along the dominant direction
        data += cone_strength * dominant_direction
        
    return data

import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Measure Vector Space Anisotropy & Cone Geometry")
    parser.add_argument("--fvecs", type=str, help="Path to .fvecs dataset (optional)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    results = []

    if args.fvecs:
        print(f"Loading {args.fvecs}...")
        data = load_fvecs(args.fvecs)
        results.append(analyze_embedding_geometry(data, name=args.fvecs))
    else:
        # Run a side-by-side demonstration
        print("Running comparative analysis on synthetic benchmarks...")
        isotropic_data = generate_synthetic_data(n=10000, d=128, cone_strength=0.0)
        anisotropic_data = generate_synthetic_data(n=10000, d=128, cone_strength=3.0)
        
        results.append(analyze_embedding_geometry(isotropic_data, "Isotropic Sphere (SIFT-like)"))
        results.append(analyze_embedding_geometry(anisotropic_data, "Anisotropic Cone (LLM-like)"))

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "=" * 80)
        print(f"{'Dataset Name':<30} | {'Mean Cosine':<12} | {'Top-10 Var %':<14} | {'Geometry'}")
        print("-" * 80)
        for r in results:
            geom_label = "⚠️ ANISOTROPIC CONE" if r["is_anisotropic"] else "✅ ISOTROPIC SPHERE"
            print(f"{r['dataset']:<30} | {r['mean_pairwise_cosine']:<12.4f} | {r['top10_singular_variance_ratio']*100:<13.1f}% | {geom_label}")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
