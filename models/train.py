"""
Unified Model Training Pipeline.
Trains:
1. Deep Multi-Task Sequence Network (PyTorch on Apple Silicon MPS)
2. Gradient Boosted Tree Ensemble (LightGBM)
Evaluates on unseen test missions and exports performance benchmarks.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import joblib

from pipeline.residuals import compute_physics_residuals, FEATURE_COLUMNS
from pipeline.windowing import (
    create_sliding_windows,
    extract_tabular_window_features,
    TelemetrySequenceDataset,
    TARGET_SENSOR_COLUMNS
)
from models.deep_sequence import UAVMultiTaskDeepNet
from models.tree_ensemble import UAVTreeEnsemble
from sim.attacks import ATTACK_NAMES, SENSOR_KEYS


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_deep_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    in_features: int,
    num_attacks: int,
    num_sensors: int,
    epochs: int = 15,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    save_path: str = "checkpoints/best_deep_model.pt"
) -> UAVMultiTaskDeepNet:
    """Trains the multi-task deep neural network with early stopping and checkpointing."""
    model = UAVMultiTaskDeepNet(
        in_features=in_features,
        num_attacks=num_attacks,
        num_sensors=num_sensors,
        hidden_dim=64,
        dropout=0.2
    ).to(device)

    criterion_att = nn.CrossEntropyLoss()
    criterion_sen = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    Path("checkpoints").mkdir(exist_ok=True)

    print(f"\n--- Training Deep Sequence Model on {device.type.upper()} ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_att_correct, total_train = 0.0, 0, 0

        for X_b, y_att_b, y_sen_b in train_loader:
            X_b = X_b.to(device)
            y_att_b = y_att_b.to(device)
            y_sen_b = y_sen_b.to(device)

            optimizer.zero_grad()
            att_logits, sen_logits = model(X_b)
            loss_att = criterion_att(att_logits, y_att_b)
            loss_sen = criterion_sen(sen_logits, y_sen_b)
            total_loss = loss_att + 1.2 * loss_sen

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += total_loss.item() * len(X_b)
            preds = torch.argmax(att_logits, dim=1)
            train_att_correct += (preds == y_att_b).sum().item()
            total_train += len(X_b)

        scheduler.step()
        train_loss /= total_train
        train_acc = train_att_correct / total_train

        # Validation
        model.eval()
        val_loss, val_att_correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_att_b, y_sen_b in val_loader:
                X_b, y_att_b, y_sen_b = X_b.to(device), y_att_b.to(device), y_sen_b.to(device)
                att_logits, sen_logits = model(X_b)
                v_loss = criterion_att(att_logits, y_att_b) + 1.2 * criterion_sen(sen_logits, y_sen_b)
                val_loss += v_loss.item() * len(X_b)
                preds = torch.argmax(att_logits, dim=1)
                val_att_correct += (preds == y_att_b).sum().item()
                total_val += len(X_b)

        val_loss /= total_val
        val_acc = val_att_correct / total_val

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            best_epoch = epoch

    print(f"Best model saved at Epoch {best_epoch} with Val Loss {best_val_loss:.4f} -> {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


import argparse
from models.export_onnx import export_to_onnx
from evaluation.report_generator import generate_research_figures


def main():
    parser = argparse.ArgumentParser(description="Train Tactical UAV Intrusion Detection Models")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs for deep model (default: 12)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training (default: 128)")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate (default: 0.0015)")
    parser.add_argument("--export-onnx", action="store_true", default=True, help="Automatically export trained model to ONNX (default: True)")
    parser.add_argument("--no-export-onnx", dest="export_onnx", action="store_false", help="Disable automatic ONNX export")
    parser.add_argument("--eval-report", action="store_true", default=True, help="Regenerate evaluation figures and confusion matrix (default: True)")
    parser.add_argument("--no-eval-report", dest="eval_report", action="store_false", help="Disable figure generation")
    args = parser.parse_args()

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    Path("evaluation").mkdir(parents=True, exist_ok=True)
    Path("weights").mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using Compute Device: {device}")
    print(f"Hyperparameters: Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}")

    # 1. Load Data
    print("Loading Parquet datasets...")
    train_raw = pd.read_parquet("data/train/train_telemetry.parquet")
    val_raw = pd.read_parquet("data/val/val_telemetry.parquet")
    test_raw = pd.read_parquet("data/test/test_telemetry.parquet")

    # 2. Extract Physics Residuals
    print("Computing invariant physics residuals across flight missions...")
    train_feat = compute_physics_residuals(train_raw)
    val_feat = compute_physics_residuals(val_raw)
    test_feat = compute_physics_residuals(test_raw)

    # 3. Create Sliding Windows
    print("Generating sliding windows (1.0s window @ 50Hz, 80% overlap)...")
    X_train_raw, y_att_train, y_sen_train = create_sliding_windows(train_feat)
    X_val_raw, y_att_val, y_sen_val = create_sliding_windows(val_feat)
    X_test_raw, y_att_test, y_sen_test = create_sliding_windows(test_feat)

    print(f"Windows generated: Train={len(X_train_raw):,}, Val={len(X_val_raw):,}, Test={len(X_test_raw):,}")

    # 4. Standardize Features across time
    N, T, F = X_train_raw.shape
    scaler = StandardScaler()
    # Reshape to (N * T, F) to fit scaler
    scaler.fit(X_train_raw.reshape(-1, F))
    joblib.dump(scaler, "checkpoints/scaler.joblib")

    X_train = scaler.transform(X_train_raw.reshape(-1, F)).reshape(N, T, F).astype(np.float32)
    X_val = scaler.transform(X_val_raw.reshape(-1, F)).reshape(len(X_val_raw), T, F).astype(np.float32)
    X_test = scaler.transform(X_test_raw.reshape(-1, F)).reshape(len(X_test_raw), T, F).astype(np.float32)

    # DataLoaders for Deep Model
    batch_size = args.batch_size
    train_dataset = TelemetrySequenceDataset(X_train, y_att_train, y_sen_train)
    val_dataset = TelemetrySequenceDataset(X_val, y_att_val, y_sen_val)
    test_dataset = TelemetrySequenceDataset(X_test, y_att_test, y_sen_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. Train Deep Sequence Model
    deep_model = train_deep_model(
        train_loader=train_loader,
        val_loader=val_loader,
        in_features=F,
        num_attacks=len(ATTACK_NAMES),
        num_sensors=len(SENSOR_KEYS),
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path="checkpoints/best_deep_model.pt"
    )

    # 6. Train Tree Ensemble Benchmark
    print("\nExtracting tabular features for Gradient Boosted Trees...")
    X_train_tab = extract_tabular_window_features(X_train)
    X_test_tab = extract_tabular_window_features(X_test)

    tree_model = UAVTreeEnsemble(n_estimators=100, max_depth=5, learning_rate=0.06)
    tree_model.fit(X_train_tab, y_att_train, y_sen_train)
    tree_model.save("checkpoints/lgb_model.joblib")

    # 7. Comprehensive Evaluation on Test Set
    print("\n================== BENCHMARK EVALUATION (UNSEEN TEST SET) ==================")
    deep_model.eval()
    all_deep_att_preds = []
    all_deep_sen_preds = []
    with torch.no_grad():
        for X_b, _, _ in test_loader:
            X_b = X_b.to(device)
            att_l, sen_l = deep_model(X_b)
            all_deep_att_preds.append(torch.argmax(att_l, dim=1).cpu().numpy())
            all_deep_sen_preds.append((torch.sigmoid(sen_l) > 0.5).cpu().numpy().astype(int))

    deep_att_preds = np.concatenate(all_deep_att_preds)
    deep_sen_preds = np.concatenate(all_deep_sen_preds)

    tree_att_preds, tree_sen_preds = tree_model.predict(X_test_tab)

    # Metrics
    deep_acc = float(accuracy_score(y_att_test, deep_att_preds))
    deep_f1 = float(f1_score(y_att_test, deep_att_preds, average="macro"))
    tree_acc = float(accuracy_score(y_att_test, tree_att_preds))
    tree_f1 = float(f1_score(y_att_test, tree_att_preds, average="macro"))

    # Sensor Exact Match Ratio (All 10 sensors correctly diagnosed)
    deep_sen_exact = float(np.mean(np.all(deep_sen_preds == y_sen_test, axis=1)))
    tree_sen_exact = float(np.mean(np.all(tree_sen_preds == y_sen_test, axis=1)))

    # Per-sensor accuracy
    sensor_accs = {}
    for i, s_name in enumerate(SENSOR_KEYS):
        s_acc = float(accuracy_score(y_sen_test[:, i], deep_sen_preds[:, i]))
        sensor_accs[s_name] = s_acc

    print(f"\n[DEEP SEQUENCE MODEL] Attack Accuracy: {deep_acc*100:.2f}% | Macro F1: {deep_f1*100:.2f}% | Sensor Exact Match: {deep_sen_exact*100:.2f}%")
    print(f"[LIGHTGBM ENSEMBLE]   Attack Accuracy: {tree_acc*100:.2f}% | Macro F1: {tree_f1*100:.2f}% | Sensor Exact Match: {tree_sen_exact*100:.2f}%")

    print("\nDetailed Attack Classification Report (Deep Sequence Model):")
    print(classification_report(y_att_test, deep_att_preds, target_names=ATTACK_NAMES, digits=4))

    # Save benchmark results
    Path("evaluation").mkdir(exist_ok=True)
    benchmarks = {
        "deep_model": {
            "attack_accuracy": deep_acc,
            "attack_macro_f1": deep_f1,
            "sensor_exact_match_ratio": deep_sen_exact,
            "per_sensor_accuracy": sensor_accs,
        },
        "lightgbm_model": {
            "attack_accuracy": tree_acc,
            "attack_macro_f1": tree_f1,
            "sensor_exact_match_ratio": tree_sen_exact,
        }
    }
    with open("evaluation/benchmark_results.json", "w") as f:
        json.dump(benchmarks, f, indent=2)

    print("Benchmark results saved to evaluation/benchmark_results.json")

    # 8. Export to Production ONNX
    if args.export_onnx:
        print("\n================== ONNX EDGE EXPORT ==================")
        export_to_onnx(
            checkpoint_path="checkpoints/best_deep_model.pt",
            output_onnx_path="weights/uav_ids_model.onnx",
            window_size=50,
            in_features=F
        )

    # 9. Regenerate Dashboard Figures
    if args.eval_report:
        print("\n================== REGENERATING DASHBOARD FIGURES ==================")
        generate_research_figures()

    print("\n✓ Full model training and deployment export finished successfully!")


if __name__ == "__main__":
    main()
