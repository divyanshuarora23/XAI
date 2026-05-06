"""
app.py - Flask backend with ML predictions + real-time alerts + dashboard
POST /predict          - File upload (PCAP/CSV) with full explanations
POST /predict-rt       - Real-time packet features (from Scapy)
GET  /alerts           - Get recent alerts
GET  /summary          - Get alert summary
GET  /frontend/*       - Serve static frontend files
"""

import os
import sys
import tempfile
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime

# Make backend/ and model/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from predict import Predictor
from backend.parsers import parse_file
from backend.database import (
    init_db, add_alert, get_recent_alerts,
    get_alerts_summary, update_statistics
)
from backend.websocket_server import init_socketio

# ── App setup ──────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

socketio, broadcast_alert, broadcast_summary_update = init_socketio(app)

ALLOWED_EXTENSIONS = {'.csv', '.pcap', '.pcapng'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize database
init_db()

# Load model once
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'artifacts')
predictor = Predictor(ARTIFACTS_DIR)

# Simple in-memory predictor for real-time (lightweight)
class RealtimePredictor:
    """Lightweight predictor for real-time packets"""
    def predict(self, features):
        # Simple heuristic for demo (replace with actual ML if needed)
        suspicious_ports = [22, 23, 3389, 445, 135, 139]  # SSH, telnet, RDP, SMB, etc.
        suspicious_flags = [16, 32]  # SYN, RST

        risk_score = 0

        if features.get('dport') in suspicious_ports:
            risk_score += 0.3
        if features.get('flags') in suspicious_flags:
            risk_score += 0.2
        if features.get('bytes', 0) > 10000:
            risk_score += 0.15

        is_anomaly = risk_score > 0.4
        confidence = min(risk_score, 1.0)

        if is_anomaly:
            threat_level = 'HIGH' if confidence > 0.7 else 'MEDIUM'
        else:
            threat_level = 'LOW'

        return {
            'prediction': 'Risk' if is_anomaly else 'Safe',
            'confidence': confidence,
            'threat_level': threat_level,
            'risk_score': risk_score
        }

rt_predictor = RealtimePredictor()

# ── Routes ────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    """File upload endpoint - full predictions with SHAP/LIME"""

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    uploaded = request.files['file']
    if uploaded.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    ext = os.path.splitext(secure_filename(uploaded.filename))[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported file type "{ext}".'}), 415

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        uploaded.save(tmp_path)

    try:
        X = parse_file(tmp_path, ARTIFACTS_DIR)
        result = predictor.predict(X)

        # Log to database
        pred = result.get('label', 'Unknown')
        conf = result.get('confidence', 0.0)
        threat = 'HIGH' if pred == 'Risk' else 'LOW'

        alert_id = add_alert(
            src_ip='file-upload',
            dst_ip='analyzer',
            src_port=0,
            dst_port=0,
            prediction=pred,
            confidence=conf,
            threat_level=threat,
            explanation='File analysis'
        )
        update_statistics(anomalies=1 if pred == 'Risk' else 0, normal=1 if pred == 'Safe' else 0)

        broadcast_alert({
            'id': alert_id,
            'src_ip': 'file-upload',
            'dst_ip': 'analyzer',
            'src_port': 0,
            'dst_port': 0,
            'prediction': pred,
            'confidence': conf,
            'threat_level': threat,
            'protocol': 'file',
            'bytes': 0,
            'packets': 1,
            'created_at': datetime.now().isoformat()
        })
        broadcast_summary_update()

        return jsonify(result), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 422
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({'error': f'Internal error: {str(e)}'}), 500
    finally:
        os.unlink(tmp_path)


@app.route('/predict-rt', methods=['POST'])
def predict_rt():
    """Real-time prediction endpoint from Scapy capture"""

    try:
        data = request.get_json()

        # Quick prediction
        result = rt_predictor.predict(data)

        # Log to database
        alert_id = add_alert(
            src_ip=data.get('src', '0.0.0.0'),
            dst_ip=data.get('dst', '0.0.0.0'),
            src_port=data.get('sport', 0),
            dst_port=data.get('dport', 0),
            prediction=result['prediction'],
            confidence=result['confidence'],
            threat_level=result['threat_level'],
            protocol='tcp' if data.get('flags') else 'udp',
            bytes_sent=data.get('bytes', 0),
            packets=data.get('packets', 1)
        )

        # Update stats
        update_statistics(
            anomalies=1 if result['prediction'] == 'Risk' else 0,
            normal=1 if result['prediction'] == 'Safe' else 0,
            threat_level=result['threat_level']
        )

        broadcast_alert({
            'id': alert_id,
            'src_ip': data.get('src', '0.0.0.0'),
            'dst_ip': data.get('dst', '0.0.0.0'),
            'src_port': data.get('sport', 0),
            'dst_port': data.get('dport', 0),
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'threat_level': result['threat_level'],
            'protocol': 'tcp' if data.get('flags') else 'udp',
            'bytes': data.get('bytes', 0),
            'packets': data.get('packets', 1),
            'created_at': datetime.now().isoformat()
        })
        broadcast_summary_update()

        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f"Real-time prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts"""
    limit = request.args.get('limit', default=50, type=int)
    alerts = get_recent_alerts(limit=limit)
    return jsonify({'alerts': alerts}), 200


@app.route('/summary', methods=['GET'])
def get_summary():
    """Get alert summary"""
    summary = get_alerts_summary()
    return jsonify(summary), 200


@app.route('/frontend/<path:filename>')
def serve_frontend(filename):
    """Serve frontend files"""
    frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
    return send_from_directory(frontend_dir, filename)


@app.route('/')
def index():
    """Redirect to dashboard"""
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), '..', 'frontend'),
        'index.html'
    )


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════
    ║     XAI IoT Firewall - Real-time IDS Dashboard               ║
    ╚═══════════════════════════════════════════════════════════════

    🌐 Web UI:        http://localhost:5000/frontend/index.html
    📊 Dashboard:     http://localhost:5000/frontend/dashboard.html
    📡 API:           http://localhost:5000/predict (file upload)
    🔴 Real-time:     http://localhost:5000/predict-rt (Scapy)

    """)
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)
