"""
preprocess.py
-------------
Handles loading, cleaning, and feature engineering for:
- UNSW-NB15 dataset (CSV)
- IoT-23 dataset (CSV)

Output: train/val/test numpy arrays ready for CNN-BiLSTM input.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib
import os

# ── Column config ────────────────────────────────────────────────────────────

# Columns to drop — identifiers, not features
DROP_COLS = []

# Target column
LABEL_COL = 'label'   # 0 = normal, 1 = attack

# Sequence window size for BiLSTM (number of consecutive rows per sample)
WINDOW_SIZE = 10

# ── Main entry point ──────────────────────────────────────────────────────────

def load_and_preprocess(csv_paths: list, save_dir: str = 'model/artifacts'):
    """
    Full preprocessing pipeline.

    Args:
        csv_paths : list of paths to UNSW-NB15 (or IoT-23) CSV files
        save_dir  : where to save scaler + encoder for inference

    Returns:
        X_train, X_val, X_test  (3D: samples × window × features)
        y_train, y_val, y_test  (1D integer arrays)
        class_weights           (dict for model training)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load
    df = _load_csvs(csv_paths)
    print(f"[preprocess] Loaded {len(df):,} rows, {df.shape[1]} columns")

    # 2. Clean
    df = _clean(df)
    print(f"[preprocess] After cleaning: {len(df):,} rows")

    # 3. Encode categoricals
    df, encoders = _encode_categoricals(df)

    # 4. Split features / label
    X = df.drop(columns=[LABEL_COL]).values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int32)

    # 5. Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    print(f"[preprocess] Scaler saved → {save_dir}/scaler.pkl")

    # 6. Build sliding windows  (samples × WINDOW_SIZE × features)
    X_seq, y_seq = _build_windows(X, y, WINDOW_SIZE)
    print(f"[preprocess] Windowed shape: X={X_seq.shape}  y={y_seq.shape}")

    # 7. Train / val / test split  (stratified)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_seq, y_seq, test_size=0.30, stratify=y_seq, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    # 8. SMOTE on training set only (2D required by SMOTE)
    n_samples, n_steps, n_feats = X_train.shape
    X_train_2d = X_train.reshape(n_samples, n_steps * n_feats)
    sm = SMOTE(random_state=42)
    X_train_2d, y_train = sm.fit_resample(X_train_2d, y_train)
    X_train = X_train_2d.reshape(-1, n_steps, n_feats)
    print(f"[preprocess] After SMOTE — train: {X_train.shape}, "
        f"class counts: {np.bincount(y_train)}")

    # 9. Class weights (fallback if SMOTE isn't enough)
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(cw))

    # 10. Save encoders
    joblib.dump(encoders, os.path.join(save_dir, 'encoders.pkl'))

    print("[preprocess] ✅ Done.")
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_csvs(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _clean(df):
    # Drop unwanted columns (ignore if missing)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Drop rows with missing label
    df = df.dropna(subset=[LABEL_COL])

    # Fill remaining NaNs with column median (numeric) or mode (categorical)
    for col in df.columns:
        if df[col].dtype == object:
            mode_vals = df[col].mode()
            df[col] = df[col].fillna(mode_vals[0] if len(mode_vals) > 0 else 'unknown')
        else:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Ensure label is binary integer
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    return df.reset_index(drop=True)


def _encode_categoricals(df):
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col == LABEL_COL:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def _build_windows(X, y, window_size):
    """
    Sliding window: each sample is `window_size` consecutive traffic rows.
    Label for the window = label of the LAST row in the window.
    """
    n = len(X)
    X_windows = np.lib.stride_tricks.sliding_window_view(
        X, (window_size, X.shape[1])
    ).squeeze(axis=1)                    # shape: (n-w+1, w, features)
    y_windows = y[window_size - 1:]     # label of last row in window
    return X_windows.astype(np.float32), y_windows.astype(np.int32)