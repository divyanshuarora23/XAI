#!/usr/bin/env python3
"""
Updated real-time packet capture with backend integration
Now sends predictions to Flask backend via /predict-rt endpoint
"""

import json
import logging
from scapy.all import sniff, IP, TCP, UDP, ICMP
import requests
from datetime import datetime
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('realtime-ids')

BACKEND_URL = 'http://127.0.0.1:5000/predict-rt'
INTERFACE = 'eth0'

def extract_features(pkt):
    """Extract features from a packet"""
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
        features['flags'] = int(pkt[TCP].flags)
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
        
        # Send to backend for real-time prediction
        response = requests.post(
            BACKEND_URL,
            json=features,
            timeout=2
        )
        
        if response.status_code == 200:
            result = response.json()
            
            prediction = result.get('prediction', 'UNKNOWN')
            confidence = result.get('confidence', 0.0)
            threat_level = result.get('threat_level', 'UNKNOWN')
            
            if prediction == 'Risk':
                logger.warning(
                    f'🚨 ANOMALY: {features["src"]}:{features["sport"]} → '
                    f'{features["dst"]}:{features["dport"]} | '
                    f'Confidence: {confidence:.2%} | Threat: {threat_level}'
                )
            else:
                logger.info(
                    f'✓ Normal: {features["src"]}:{features["sport"]} → '
                    f'{features["dst"]}:{features["dport"]}'
                )
        else:
            logger.error(f'Backend error: {response.status_code}')
    
    except requests.exceptions.Timeout:
        pass  # Silently skip timeouts for performance
    except requests.exceptions.RequestException as e:
        logger.error(f'Connection error: {e}')
    except Exception as e:
        logger.error(f'Error: {e}')

def main():
    logger.info(f'Starting real-time capture on {INTERFACE}...')
    logger.info(f'Backend: {BACKEND_URL}')
    logger.info('Press Ctrl+C to stop.')
    
    try:
        sniff(
            iface=INTERFACE,
            prn=packet_callback,
            store=False,
            filter='tcp or udp or icmp',
        )
    except KeyboardInterrupt:
        logger.info('Stopping capture.')
    except PermissionError:
        logger.error('❌ You need root/sudo privileges!')
        sys.exit(1)

if __name__ == '__main__':
    main()
