import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings('ignore')

def main():
    print("=" * 65)
    print("      LIGHTGBM BENCHMARK ON GCP COMPUTE NODE (CPU)")
    print("=" * 65)

    # 1. Load Data
    data_path = os.path.expanduser("~/ml-benchmark/creditcard.csv")
    print(f"[1/5] Loading dataset from {data_path}...")
    t0 = time.time()
    df = pd.read_csv(data_path)
    data_load_time = time.time() - t0
    print(f"      Dataset loaded in {data_load_time:.4f} seconds ({len(df)} rows, {df.shape[1]} columns).")

    # 2. Train / Test Split
    print("[2/5] Preprocessing and splitting data (80% train, 20% test)...")
    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Model Training
    print("[3/5] Training LightGBM Classifier (100 trees)...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )

    t_train_start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc'
    )
    training_time = time.time() - t_train_start
    best_iter = getattr(model, 'best_iteration_', 100)
    best_iteration = best_iter if (best_iter is not None and best_iter > 0) else 100
    print(f"      Training completed in {training_time:.4f} seconds. Best iteration: {best_iteration}")

    # 4. Evaluation
    print("[4/5] Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_roc = float(roc_auc_score(y_test, y_proba))
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))

    # 5. Inference Benchmarking
    print("[5/5] Measuring inference latency and throughput...")
    
    # Single row latency (average over 1000 iterations)
    sample_single = X_test.iloc[[0]]
    for _ in range(50):
        _ = model.predict(sample_single)
    
    latencies = []
    for _ in range(1000):
        t_s = time.perf_counter()
        _ = model.predict(sample_single)
        t_e = time.perf_counter()
        latencies.append((t_e - t_s) * 1000.0)
    
    inference_latency_ms = float(np.mean(latencies))

    # Batch throughput (1000 rows)
    sample_batch_1000 = X_test.iloc[:1000]
    for _ in range(10):
        _ = model.predict(sample_batch_1000)
    
    batch_times = []
    for _ in range(50):
        t_s = time.perf_counter()
        _ = model.predict(sample_batch_1000)
        t_e = time.perf_counter()
        batch_times.append(t_e - t_s)
    
    avg_batch_time = float(np.mean(batch_times))
    inference_throughput = float(1000.0 / avg_batch_time)

    # Output dictionary
    results = {
        "data_loading_time_sec": round(data_load_time, 4),
        "training_time_sec": round(training_time, 4),
        "best_iteration": int(best_iteration),
        "auc_roc": round(auc_roc, 6),
        "accuracy": round(accuracy, 6),
        "f1_score": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "inference_latency_single_ms": round(inference_latency_ms, 4),
        "inference_throughput_1000_samples_per_sec": round(inference_throughput, 2)
    }

    # Save to JSON
    output_json_path = os.path.expanduser("~/ml-benchmark/benchmark_result.json")
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "=" * 65)
    print("               BENCHMARK RESULTS TABLE")
    print("=" * 65)
    print(f"| {'Metric':<35} | {'Value':<20} |")
    print(f"|{'-'*37}|{'-'*22}|")
    print(f"| {'Thời gian load data':<35} | {data_load_time:.4f} s{'':<13} |")
    print(f"| {'Thời gian training':<35} | {training_time:.4f} s{'':<13} |")
    print(f"| {'Best iteration':<35} | {best_iteration:<20} |")
    print(f"| {'AUC-ROC':<35} | {auc_roc:.6f}{'':<12} |")
    print(f"| {'Accuracy':<35} | {accuracy:.6f}{'':<12} |")
    print(f"| {'F1-Score':<35} | {f1:.6f}{'':<12} |")
    print(f"| {'Precision':<35} | {precision:.6f}{'':<12} |")
    print(f"| {'Recall':<35} | {recall:.6f}{'':<12} |")
    print(f"| {'Inference latency (1 row)':<35} | {inference_latency_ms:.4f} ms{'':<12} |")
    print(f"| {'Inference throughput (1000 rows)':<35} | {inference_throughput:.2f} rows/s{'':<7} |")
    print("=" * 65)
    print(f"\n[SUCCESS] Results saved to {output_json_path}\n")

if __name__ == "__main__":
    main()
