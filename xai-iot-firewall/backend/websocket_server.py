"""
WebSocket server for real-time alert streaming
Integrates with Flask backend to push alerts to connected clients
"""

from flask import Blueprint, render_template_string, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import logging
from datetime import datetime
from backend.database import get_recent_alerts, get_alerts_summary, add_alert, get_statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('websocket-server')

# Create blueprint
ws_bp = Blueprint('websocket', __name__)


def init_socketio(app):
    """Initialize SocketIO with Flask app"""
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        ping_timeout=60,
        ping_interval=25,
        logger=True,
        engineio_logger=False
    )

    # Track connected clients
    connected_clients = set()

    @socketio.on('connect')
    def handle_connect():
        """Client connects"""
        client_id = request.sid
        connected_clients.add(client_id)
        logger.info(f'✓ Client connected: {client_id} (Total: {len(connected_clients)})')

        # Send recent alerts on connect
        alerts = get_recent_alerts(limit=20)
        summary = get_alerts_summary()

        emit('initial_data', {
            'alerts': alerts,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })

        # Notify all clients of connection
        emit('client_count', {'count': len(connected_clients)}, broadcast=True)

    @socketio.on('disconnect')
    def handle_disconnect():
        """Client disconnects"""
        client_id = request.sid
        connected_clients.discard(client_id)
        logger.info(f'✗ Client disconnected: {client_id} (Total: {len(connected_clients)})')
        emit('client_count', {'count': len(connected_clients)}, broadcast=True)

    @socketio.on('request_alerts')
    def handle_alert_request(data):
        """Client requests recent alerts"""
        limit = data.get('limit', 50)
        alerts = get_recent_alerts(limit=limit)
        emit('alert_update', {'alerts': alerts})

    @socketio.on('request_summary')
    def handle_summary_request():
        """Client requests summary statistics"""
        summary = get_alerts_summary()
        stats = get_statistics()
        emit('summary_update', {'summary': summary, 'stats': stats})

    def broadcast_alert(alert_data):
        """Broadcast new alert to all connected clients"""
        socketio.emit('new_alert', alert_data, broadcast=True)
        logger.info(f'📢 Alert broadcasted to {len(connected_clients)} clients')

   def broadcast_alert(alert_data):
        """Broadcast new alert to all connected clients"""
        # FIX: Remove broadcast=True. socketio.emit() broadcasts by default.
        socketio.emit('new_alert', alert_data) 
        logger.info(f'📢 Alert broadcasted to {len(connected_clients)} clients')

    def broadcast_summary_update():
        """Broadcast summary update"""
        summary = get_alerts_summary()
        # FIX: Remove broadcast=True
        socketio.emit('summary_update', {'summary': summary})

    return socketio, broadcast_alert, broadcast_summary_update


# Export functions for use in app.py
__all__ = ['init_socketio', 'ws_bp']
