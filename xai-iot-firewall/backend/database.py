"""
SQLite database for storing alerts and predictions
"""

import sqlite3
from datetime import datetime
from contextlib import contextmanager
import os

DB_PATH = '/workspace/alerts.db'

def init_db():
    """Initialize database with alerts table"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Alerts table
    c.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            src_ip TEXT,
            dst_ip TEXT,
            src_port INTEGER,
            dst_port INTEGER,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            threat_level TEXT,
            protocol TEXT,
            bytes INTEGER,
            packets INTEGER,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Statistics table
    c.execute('''
        CREATE TABLE IF NOT EXISTS statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_packets INTEGER DEFAULT 0,
            total_anomalies INTEGER DEFAULT 0,
            total_normal INTEGER DEFAULT 0,
            high_threat_count INTEGER DEFAULT 0,
            medium_threat_count INTEGER DEFAULT 0,
            low_threat_count INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Initialize stats if empty
    c.execute('SELECT COUNT(*) FROM statistics')
    if c.fetchone()[0] == 0:
        c.execute('''
            INSERT INTO statistics 
            (total_packets, total_anomalies, total_normal, high_threat_count, medium_threat_count, low_threat_count)
            VALUES (0, 0, 0, 0, 0, 0)
        ''')
    
    conn.commit()
    conn.close()

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def add_alert(src_ip, dst_ip, src_port, dst_port, prediction, confidence, threat_level, protocol='tcp', bytes_sent=0, packets=1, explanation=''):
    """Add an alert to the database"""
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO alerts 
            (timestamp, src_ip, dst_ip, src_port, dst_port, prediction, confidence, threat_level, protocol, bytes, packets, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            src_ip,
            dst_ip,
            src_port,
            dst_port,
            prediction,
            confidence,
            threat_level,
            protocol,
            bytes_sent,
            packets,
            explanation
        ))
        conn.commit()
        return c.lastrowid

def get_recent_alerts(limit=50):
    """Get recent alerts"""
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''
            SELECT * FROM alerts 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in c.fetchall()]

def get_alerts_by_threat_level(threat_level, limit=50):
    """Get alerts by threat level"""
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''
            SELECT * FROM alerts 
            WHERE threat_level = ?
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (threat_level, limit))
        return [dict(row) for row in c.fetchall()]

def get_statistics():
    """Get current statistics"""
    with get_db() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM statistics LIMIT 1')
        row = c.fetchone()
        return dict(row) if row else None

def update_statistics(anomalies=0, normal=0, threat_level=None):
    """Update statistics after prediction"""
    with get_db() as conn:
        c = conn.cursor()
        
        # Update totals
        c.execute('UPDATE statistics SET total_packets = total_packets + 1')
        
        if anomalies > 0:
            c.execute('UPDATE statistics SET total_anomalies = total_anomalies + ?', (anomalies,))
        if normal > 0:
            c.execute('UPDATE statistics SET total_normal = total_normal + ?', (normal,))
        
        # Update threat level counts
        if threat_level == 'HIGH':
            c.execute('UPDATE statistics SET high_threat_count = high_threat_count + 1')
        elif threat_level == 'MEDIUM':
            c.execute('UPDATE statistics SET medium_threat_count = medium_threat_count + 1')
        elif threat_level == 'LOW':
            c.execute('UPDATE statistics SET low_threat_count = low_threat_count + 1')
        
        c.execute('UPDATE statistics SET updated_at = CURRENT_TIMESTAMP')
        conn.commit()

def get_alerts_summary():
    """Get summary stats"""
    with get_db() as conn:
        c = conn.cursor()
        c.execute('SELECT COUNT(*) as total FROM alerts')
        total = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) as high FROM alerts WHERE threat_level = 'HIGH'")
        high = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) as medium FROM alerts WHERE threat_level = 'MEDIUM'")
        medium = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) as low FROM alerts WHERE threat_level = 'LOW'")
        low = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) as anomalies FROM alerts WHERE prediction = 'Anomaly'")
        anomalies = c.fetchone()[0]
        
        return {
            'total_alerts': total,
            'high_threat': high,
            'medium_threat': medium,
            'low_threat': low,
            'total_anomalies': anomalies
        }

def get_alerts_by_ip(ip_address, limit=50):
    """Get alerts involving specific IP"""
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''
            SELECT * FROM alerts 
            WHERE src_ip = ? OR dst_ip = ?
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (ip_address, ip_address, limit))
        return [dict(row) for row in c.fetchall()]

# Initialize on import
if not os.path.exists(DB_PATH):
    init_db()
