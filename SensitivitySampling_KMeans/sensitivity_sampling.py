import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from get_data import X_cover, X_census, X_tower


# -----------------------------
# Dataset Utilities
# -----------------------------
def load_dataset(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    print(f"Loading dataset: {file_path}")
    if ext == '.csv':
        df = pd.read_csv(file_path)
        df = df.select_dtypes(include=[np.number])
        return df.to_numpy(dtype=np.float64)
    elif ext == '.txt':
        return np.loadtxt(file_path, delimiter=",", dtype=np.float64)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def standardize(data):
    return StandardScaler().fit_transform(data)

# -----------------------------
# Sensitivity Sampling
# -----------------------------
def compute_sensitivities(data, k):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=1, random_state=42)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    cost = np.sum((data - centers[labels]) ** 2, axis=1)
    total_cost = np.sum(cost)
    sensitivities = cost / total_cost
    return sensitivities, total_cost, centers

def sample_coreset(data, sensitivities, total_cost, T):
    probabilities = sensitivities / np.sum(sensitivities)
    indices = np.random.choice(len(data), size=T, p=probabilities)
    coreset = data[indices]
    weights = 1 / (T * probabilities[indices])
    return coreset, weights

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_kmeans_cost(data, k, sample_weights=None, centers=None):
    if centers is not None:
        labels = np.argmin(np.linalg.norm(data[:, None] - centers[None, :], axis=2), axis=1)
        dists = np.sum((data - centers[labels]) ** 2, axis=1)
        return np.sum(dists * (sample_weights if sample_weights is not None else 1))
    else:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=1, random_state=42)
        kmeans.fit(data, sample_weight=sample_weights)
        return kmeans.inertia_

# -----------------------------
# Distortion Analysis with Timing
# -----------------------------
def evaluate_coreset_distortion(data, k_values, coreset_fractions, num_runs=5):
    results = {}

    for frac in coreset_fractions:
        T = int(frac * len(data))
        print(f"\n----- Coreset Fraction: {int(frac * 100)}% | Size: {T} -----")
        distortions = []
        start_total = time.time()

        for k in k_values:
            print(f"Evaluating for k = {k}")
            start_k = time.time()

            sensitivities, total_cost, centers = compute_sensitivities(data, k)
            full_cost = evaluate_kmeans_cost(data, k, centers=centers)

            max_distortions = []
            for _ in range(num_runs):
                coreset, weights = sample_coreset(data, sensitivities, total_cost, T)
                coreset_cost = evaluate_kmeans_cost(coreset, k, sample_weights=weights, centers=centers)
                distortion = max(coreset_cost / full_cost, full_cost / coreset_cost)
                max_distortions.append(distortion)

            max_dist = np.max(max_distortions)
            distortions.append(max_dist)
            print(f"Max Distortion for k={k}: {max_dist:.4f} | Time: {time.time() - start_k:.2f}s")

        results[int(frac * 100)] = {
            'distortions': distortions,
            'time': time.time() - start_total
        }
        print(f"✅ Done for {int(frac * 100)}% | Total Time: {results[int(frac * 100)]['time']:.2f}s")

    return results

# -----------------------------
# Plotting and Saving
# -----------------------------
def plot_and_save_results(k_values, results, dataset_name, result_dir="results"):
    os.makedirs(result_dir, exist_ok=True)
    fig, ax = plt.subplots()
    report_lines = []

    for frac, data in results.items():
        ax.plot(k_values, data['distortions'], marker='o', label=f"{frac}%")
        line = f"Coreset {frac}%: Max distortions = {data['distortions']} | Time = {data['time']:.2f}s"
        report_lines.append(line)

    ax.set_xlabel("k (Number of Clusters)")
    ax.set_ylabel("Max Distortion")
    ax.set_title(f"Sensitivity Sampling Distortion - {dataset_name}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plot_path = os.path.join(result_dir, f"{dataset_name}_distortion.png")
    fig.savefig(plot_path)
    plt.close()

    # Save report
    report_path = os.path.join(result_dir, f"{dataset_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write("Coreset Evaluation Summary:\n")
        f.write("\n".join(report_lines))
        f.write("\n")

# -----------------------------
# Main Pipeline
# -----------------------------
def run_on_dataset(file_path, k_values=[10, 20, 30, 40, 50], coreset_fractions=[0.05, 0.10, 0.12, 0.15, 0.20]):
    raw_data = load_dataset(file_path)
    data = standardize(raw_data)
    results = evaluate_coreset_distortion(data, k_values, coreset_fractions)
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    plot_and_save_results(k_values, results, dataset_name)

# -----------------------------
# Run All Datasets in /data
# -----------------------------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # List of (dataset_name, data_matrix) tuples
    datasets = [
        ("covertype", X_cover),
        ("census", X_census)
    ]

    # First run on built-in datasets
    for name, raw_data in datasets:
        print(f"\nRunning evaluation on: {name}")
        data = standardize(raw_data)
        results = evaluate_coreset_distortion(data, k_values=[10, 20, 30, 40, 50],
                                              coreset_fractions=[0.05, 0.10, 0.12, 0.15, 0.20])
        plot_and_save_results([10, 20, 30, 40, 50], results, dataset_name=name)

    # Now process tower.txt (and other local .csv/.txt files)
    dataset_folder = "data"
    files = [os.path.join(dataset_folder, f)
             for f in os.listdir(dataset_folder)
             if f.endswith(".csv") or f.endswith(".txt")]

    for file_path in files:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        if dataset_name in ["census", "covertype"]:
            continue  # already processed above
        run_on_dataset(file_path)
