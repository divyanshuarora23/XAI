"""
train.py
--------
Trains the CNN-BiLSTM model and saves:
- best_model.keras       (full model)
- training_history.png   (loss + metric curves)
- evaluation_report.txt  (test set metrics)
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay)
import os, json

from preprocess import load_and_preprocess
from model import build_cnn_bilstm, get_callbacks


# ── Config ────────────────────────────────────────────────────────────────────

DATASET_PATHS = [
    'analysis_output/balanced_dataset.csv'
]

ARTIFACTS_DIR  = 'model/artifacts'
EPOCHS         = 50
BATCH_SIZE     = 256    # change to 2048 when on gpu
DROPOUT_RATE   = 0.3


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Preprocess
    print("\n── Phase 1: Preprocessing ──────────────────────────────────")
    (X_train, X_val, X_test, y_train, y_val, y_test, class_weights) = load_and_preprocess(DATASET_PATHS, ARTIFACTS_DIR)
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Input shape : {input_shape}")
    print(f"Train size  : {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
    print(f"Class weights: {class_weights}")

    # 2. Build model
    print("\n── Phase 2: Building Model ─────────────────────────────────")
    model = build_cnn_bilstm(input_shape, DROPOUT_RATE)
    model.summary()

    # 3. Train
    print("\n── Phase 3: Training ───────────────────────────────────────")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(os.path.join(ARTIFACTS_DIR, 'best_model.keras')),
        verbose=1,
    )

    # Save history as JSON (useful for later analysis)
    with open(os.path.join(ARTIFACTS_DIR, 'history.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals]
                for k, vals in history.history.items()}, f, indent=2)

    # 4. Plot training curves
    print("\n── Phase 4: Plotting Curves ────────────────────────────────")
    _plot_history(history, ARTIFACTS_DIR)

    # 5. Evaluate on test set
    print("\n── Phase 5: Evaluation ─────────────────────────────────────")
    _evaluate(model, X_test, y_test, ARTIFACTS_DIR)

    print(f"\n✅ Training complete. All artifacts saved to → {ARTIFACTS_DIR}/")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _plot_history(history, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CNN-BiLSTM Training History', fontsize=16, fontweight='bold')

    metrics = [
        ('loss',     'val_loss',     'Loss'),
        ('accuracy', 'val_accuracy', 'Accuracy'),
        ('auc',      'val_auc',      'AUC-ROC'),
        ('recall',   'val_recall',   'Recall'),
    ]

    for ax, (train_m, val_m, title) in zip(axes.flat, metrics):
        if train_m in history.history:
            ax.plot(history.history[train_m],  label='Train', linewidth=2)
            ax.plot(history.history[val_m],    label='Val',   linewidth=2, linestyle='--')
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Training curves saved → {path}")


def _evaluate(model, X_test, y_test, save_dir):
    y_prob = model.predict(X_test, batch_size=256, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])
    auc    = roc_auc_score(y_test, y_prob)

    print(report)
    print(f"AUC-ROC: {auc:.4f}")

    # Save report
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
        f.write(f"\nAUC-ROC: {auc:.4f}\n")
    print(f"  Report saved → {report_path}")

    # Confusion matrix plot
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix — Test Set')
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {cm_path}")


if __name__ == '__main__':
    main()