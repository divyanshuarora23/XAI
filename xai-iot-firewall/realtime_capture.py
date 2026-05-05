#!/usr/bin/env python3
"""
Real-time packet capture using Scapy + feature extraction for IoT-23 dataset
Sends predictions to Flask backend running on localhost:5000
"""

import json
import logging
from scapy.all import sniff, IP, TCP, UDP, ICMP
import requests
from datetime import datetime
import sys

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('realtime-ids')

BACKEND_URL = 'http://127.0.0.1:5000/predict'
INTERFACE = 'eth0'  # Change to your interface (or 'any' for all)

def extract_features(pkt):
    """Extract IoT-23 features from a packet"""
    features = {
        'timestamp': datetime.now().isoformat(),
        'proto': 0,
        'src': '0.0.0.0',
        'dst': '0.0.0.0',
        'sport': 0,
        'dport': 0,
        'duration': 0.0,
        'bytes': len(pkt),
        'packets': 1,
        'flags': 0,
    }
    
    if IP in pkt:
        features['src'] = pkt[IP].src
        features['dst'] = pkt[IP].dst
        features['proto'] = pkt[IP].proto
        
    if TCP in pkt:
        features['sport'] = pkt[TCP].sport
        features['dport'] = pkt[TCP].dport
        features['flags'] = pkt[TCP].flags
    elif UDP in pkt:
        features['sport'] = pkt[UDP].sport
        features['dport'] = pkt[UDP].dport
    elif ICMP in pkt:
        features['sport'] = pkt[ICMP].type
        features['dport'] = pkt[ICMP].code
    
    return features

def packet_callback(pkt):
    """Callback for each captured packet"""
    try:
        features = extract_features(pkt)
        
        # Send to backend for prediction
        response = requests.post(
            BACKEND_URL,
            json=features,
            timeout=2
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Log prediction
            prediction = result.get('prediction', 'UNKNOWN')
            confidence = result.get('confidence', 0.0)
            threat_level = result.get('threat_level', 'UNKNOWN')
            
            if prediction == 'Anomaly':
                logger.warning(
                    f'🚨 ANOMALY DETECTED: '
                    f'{features["src"]}:{features["sport"]} → {features["dst"]}:{features["dport"]} | '
                    f'Confidence: {confidence:.2%} | Threat: {threat_level}'
                )
            else:
                logger.info(
                    f'✓ Normal: {features["src"]}:{features["sport"]} → {features["dst"]}:{features["dport"]}'
                )
        else:
            logger.error(f'Backend error: {response.status_code}')
    
    except requests.exceptions.RequestException as e:
        logger.error(f'Connection error: {e}')
    except Exception as e:
        logger.error(f'Error processing packet: {e}')

def main():
    logger.info(f'Starting real-time capture on {INTERFACE}...')
    logger.info(f'Backend: {BACKEND_URL}')
    logger.info('Press Ctrl+C to stop.')
    
    try:
        sniff(
            iface=INTERFACE,
            prn=packet_callback,
            store=False,
            filter='tcp or udp or icmp',  # Filter only these protocols
        )
    except KeyboardInterrupt:
        logger.info('Stopping capture...')
    except PermissionError:
        logger.error('❌ You need root/sudo privileges to capture packets!')
        sys.exit(1)

if __name__ == '__main__':
    main()
