"""
predict.py
----------
Handles inference for the Flask backend.
Accepts a preprocessed numpy array and returns:
- label      : 'Safe' or 'Risk'
- confidence : float 0–1
- shap       : explanation dict
- lime       : explanation dict
"""

import numpy as np
import tensorflow as tf
import joblib
import os

from explain import (build_shap_explainer, get_shap_explanation,
                    build_lime_explainer, get_lime_explanation)


class Predictor:
    """
    Loads the trained model + artifacts once at server startup.
    Call .predict(X) for each incoming file.
    """

    def __init__(self, artifacts_dir: str = 'model/artifacts'):
        model_path   = os.path.join(artifacts_dir, 'best_model.keras')
        scaler_path  = os.path.join(artifacts_dir, 'scaler.pkl')
        bg_path      = os.path.join(artifacts_dir, 'shap_background.npy')
        train_path   = os.path.join(artifacts_dir, 'X_train_sample.npy')
        fnames_path  = os.path.join(artifacts_dir, 'feature_names.npy')

        print("[predictor] Loading model …")
        self.model = tf.keras.models.load_model(model_path)

        print("[predictor] Loading scaler …")
        self.scaler = joblib.load(scaler_path)

        print("[predictor] Loading feature names …")
        self.feature_names = np.load(fnames_path, allow_pickle=True).tolist()

        # Background samples for SHAP (saved during training)
        print("[predictor] Building SHAP explainer …")
        bg = np.load(bg_path)
        self.shap_explainer = build_shap_explainer(self.model, bg)

        # Training sample for LIME
        print("[predictor] Building LIME explainer …")
        X_train_sample = np.load(train_path)
        self.lime_explainer = build_lime_explainer(
            X_train_sample, self.feature_names)

        print("[predictor] ✅ Ready.")

    def predict(self, X: np.ndarray) -> dict:
        """
        Args:
            X : numpy array shape (n_rows, n_features) — raw, unscaled, flat
                This comes straight from the file parser (csv_parser / pcap_parser)

        Returns:
            {
            'label'      : 'Safe' | 'Risk',
            'confidence' : float,
            'shap'       : { values, plot_base64 },
            'lime'       : { values, plot_base64 },
            }
        """
        # 1. Scale
        X_scaled = self.scaler.transform(X)

        # 2. Build sliding windows
        from preprocess import _build_windows, WINDOW_SIZE
        X_seq, _ = _build_windows(X_scaled,
                                np.zeros(len(X_scaled), dtype=np.int32),
                                WINDOW_SIZE)

        if len(X_seq) == 0:
            raise ValueError(
                f"File too short — need at least {WINDOW_SIZE} rows of traffic data.")

        # 3. Predict (average probability across all windows in the file)
        probs = self.model.predict(X_seq, verbose=0).flatten()
        mean_prob = float(probs.mean())
        label     = 'Risk' if mean_prob >= 0.5 else 'Safe'

        # 4. Explain using the window with highest probability (most suspicious)
        worst_idx = int(probs.argmax())
        worst_window = X_seq[worst_idx:worst_idx + 1]   # shape (1, w, f)

        shap_result = get_shap_explanation(
            self.shap_explainer, worst_window, self.feature_names)
        lime_result = get_lime_explanation(
            self.lime_explainer, self.model, worst_window, self.feature_names)

        return {
            'label':      label,
            'confidence': round(mean_prob, 4),
            'shap':       shap_result,
            'lime':       lime_result,
        }