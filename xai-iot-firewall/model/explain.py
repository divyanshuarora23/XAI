"""
explain.py
----------
Generates SHAP and LIME explanations for a given prediction.

Both explainers expect a 2D input (samples × features) — we flatten
the (window, features) tensor before explaining, which is standard
practice for sequence models used in tabular-origin data.
"""

import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for Flask
import io, base64, os


# ── SHAP ──────────────────────────────────────────────────────────────────────

def build_shap_explainer(model, X_background: np.ndarray):
    """
    Build a KernelSHAP explainer.

    Args:
        model        : trained Keras model
        X_background : small representative sample (100–200 rows) from training set
                    shape: (n, window_size, n_features)

    Returns:
        shap.KernelExplainer
    """
    # Flatten window dim for SHAP
    bg_2d = _flatten(X_background)

    def predict_fn(x_2d):
        x_3d = _unflatten(x_2d, model.input_shape)
        return model.predict(x_3d, verbose=0).flatten()

    explainer = shap.KernelExplainer(predict_fn, bg_2d)
    return explainer


def get_shap_explanation(explainer, sample: np.ndarray,
                        feature_names: list) -> dict:
    """
    Compute SHAP values for a single sample.

    Args:
        explainer     : KernelExplainer built by build_shap_explainer()
        sample        : shape (1, window_size, n_features)  OR  (window_size, n_features)
        feature_names : list of original feature names

    Returns:
        dict with keys:
        'values'       : list of (feature_name, shap_value) sorted by |value|
        'plot_base64'  : base64-encoded PNG of the bar chart
    """
    if sample.ndim == 2:
        sample = sample[np.newaxis, ...]   # → (1, w, f)

    sample_2d = _flatten(sample)           # → (1, w*f)
    shap_vals  = explainer.shap_values(sample_2d, nsamples=100)

    # Average SHAP values across the window dimension per feature
    n_features = len(feature_names)
    window_size = sample_2d.shape[1] // n_features
    shap_per_feature = shap_vals[0].reshape(window_size, n_features).mean(axis=0)

    pairs = sorted(zip(feature_names, shap_per_feature.tolist()),
                key=lambda x: abs(x[1]), reverse=True)

    top_n = pairs[:15]
    plot_b64 = _shap_bar_chart(top_n)

    return {
        'values':      top_n,
        'plot_base64': plot_b64,
    }


# ── LIME ──────────────────────────────────────────────────────────────────────

def build_lime_explainer(X_train: np.ndarray, feature_names: list):
    # X_train may be 2D (n, features) or 3D (n, window, features)
    # LIME always works in 2D — use features only, no window flattening
    if X_train.ndim == 3:
        train_2d = X_train[:, 0, :]   # take first timestep only
    else:
        train_2d = X_train

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train_2d,
        feature_names=feature_names,
        class_names=['Normal', 'Attack'],
        mode='classification',
        discretize_continuous=False,
        random_state=42,
    )
    return explainer


def get_lime_explanation(explainer, model, sample: np.ndarray,
                        feature_names: list) -> dict:
    """
    Generate LIME explanation for a single sample.

    Returns:
        dict with keys:
        'values'      : list of (feature_condition, weight) sorted by |weight|
        'plot_base64' : base64-encoded PNG
    """
    if sample.ndim == 2:
        sample = sample[np.newaxis, ...]

    # Take middle timestep as representative 1D sample for LIME
    if sample.ndim == 3:
        sample_2d = sample[0, sample.shape[1] // 2, :]  # shape (n_features,)
    else:
        sample_2d = sample[0]

    _, window_size, n_features = model.input_shape

    def predict_fn(x_2d):
        # x_2d shape: (n_samples, n_features) — tile across window
        x_3d = np.stack([x_2d] * window_size, axis=1)  # (n, window, features)
        probs = model.predict(x_3d, verbose=0).flatten()
        return np.column_stack([1 - probs, probs])

    exp = explainer.explain_instance(
        data_row=sample_2d,
        predict_fn=predict_fn,
        num_features=15,
        num_samples=300,
    )

    lime_vals = exp.as_list()   # [(condition_string, weight), ...]
    plot_b64  = _lime_bar_chart(lime_vals)

    return {
        'values':      lime_vals,
        'plot_base64': plot_b64,
    }


# ── Charting helpers ──────────────────────────────────────────────────────────

def _shap_bar_chart(pairs: list) -> str:
    names  = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('SHAP Value  (positive → towards Attack)', fontsize=10)
    ax.set_title('SHAP Feature Importance', fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return _fig_to_base64(fig)


def _lime_bar_chart(pairs: list) -> str:
    pairs_sorted = sorted(pairs, key=lambda x: x[1])
    names  = [p[0] for p in pairs_sorted]
    values = [p[1] for p in pairs_sorted]
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('LIME Weight  (positive → Attack class)', fontsize=10)
    ax.set_title('LIME Local Explanation', fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return _fig_to_base64(fig)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── Internal utilities ────────────────────────────────────────────────────────

def _flatten(X: np.ndarray) -> np.ndarray:
    """(n, window, features) → (n, window*features)"""
    return X.reshape(X.shape[0], -1)


def _unflatten(X_2d: np.ndarray, input_shape) -> np.ndarray:
    """(n, window*features) → (n, window, features)"""
    _, window_size, n_features = input_shape
    return X_2d.reshape(X_2d.shape[0], window_size, n_features)


def _flat_feature_names(feature_names: list, window_size: int) -> list:
    """Create names like 'proto_t0', 'proto_t1', … for each time step."""
    return [f'{f}_t{t}' for t in range(window_size) for f in feature_names]