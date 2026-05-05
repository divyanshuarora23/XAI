"""
app.py
------
Flask backend — single endpoint:
POST /predict   multipart/form-data  field: 'file'

Response JSON:
{
    "label"      : "Safe" | "Risk",
    "confidence" : 0.87,
    "shap"       : { "values": [...], "plot_base64": "..." },
    "lime"       : { "values": [...], "Kplot_base64": "..." }
}
"""

import os
import sys
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Make model/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from predict import Predictor
from parsers import parse_file

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)   # allow requests from the frontend (different port in dev)

ALLOWED_EXTENSIONS = {'.csv', '.pcap', '.pcapng'}
MAX_CONTENT_LENGTH  = 50 * 1024 * 1024   # 50 MB limit
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Load model once at startup
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'artifacts')
predictor     = Predictor(ARTIFACTS_DIR)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    # ── Validate upload ───────────────────────────────────────────────────────
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Use field name "file".'}), 400

    uploaded = request.files['file']
    if uploaded.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    ext = os.path.splitext(secure_filename(uploaded.filename))[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            'error': f'Unsupported file type "{ext}". '
                    f'Please upload a .csv or .pcap file.'
        }), 415

    # ── Save to temp file ─────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        uploaded.save(tmp_path)

    try:
        # ── Parse ─────────────────────────────────────────────────────────────
        X = parse_file(tmp_path, ARTIFACTS_DIR)

        # ── Predict + Explain ─────────────────────────────────────────────────
        result = predictor.predict(X)

        return jsonify(result), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 422

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({'error': f'Internal error: {str(e)}'}), 500

    finally:
        os.unlink(tmp_path)   # always clean up temp file


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)