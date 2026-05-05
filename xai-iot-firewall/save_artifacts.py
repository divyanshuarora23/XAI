"""
save_artifacts.py
-----------------
Run this ONCE after training is complete.
Saves the extra artifacts that predict.py and parsers.py need at inference time:
- feature_names.npy      list of column names in training order
- shap_background.npy    200 random training windows for KernelSHAP
- X_train_sample.npy     500 random training windows for LIME
"""

import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))
from preprocess import load_and_preprocess, DROP_COLS, LABEL_COL

DATASET_PATHS = [
    '../datasets/IoT_Intrusion.csv',
]
ARTIFACTS_DIR = 'model/artifacts'

def main():
    (X_train, X_val, X_test,
    y_train, y_val, y_test,
    _) = load_and_preprocess(DATASET_PATHS, ARTIFACTS_DIR)

    # Feature names: reload original CSV to get column order
    import pandas as pd
    df = pd.read_csv(DATASET_PATHS[0], low_memory=False, nrows=1)
    drop = [c for c in DROP_COLS + [LABEL_COL] if c in df.columns]
    feature_names = [c for c in df.columns if c not in drop]

    np.save(os.path.join(ARTIFACTS_DIR, 'feature_names.npy'),
            np.array(feature_names, dtype=object))
    print(f"[artifacts] Saved feature_names ({len(feature_names)} features)")

    # SHAP background: 200 random samples
    idx = np.random.choice(len(X_train), size=min(200, len(X_train)), replace=False)
    np.save(os.path.join(ARTIFACTS_DIR, 'shap_background.npy'), X_train[idx])
    print(f"[artifacts] Saved shap_background: {X_train[idx].shape}")

    # LIME training sample: 500 random samples
    idx2 = np.random.choice(len(X_train), size=min(500, len(X_train)), replace=False)
    np.save(os.path.join(ARTIFACTS_DIR, 'X_train_sample.npy'), X_train[idx2])
    print(f"[artifacts] Saved X_train_sample: {X_train[idx2].shape}")

    print("\n✅ All artifacts saved. You can now run the Flask backend.")

if __name__ == '__main__':
    main()