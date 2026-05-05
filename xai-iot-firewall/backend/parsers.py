"""
parsers.py
----------
Converts uploaded files (CSV or PCAP) into a raw feature DataFrame
with the same columns the model was trained on.
"""

import numpy as np
import pandas as pd
import joblib
import os


# Columns that must exist after parsing (same as training, minus label)
# This list is auto-populated from saved artifacts at runtime
_REQUIRED_COLS = None


def parse_file(filepath: str, artifacts_dir: str = 'model/artifacts') -> np.ndarray:
    """
    Dispatch to the correct parser based on file extension.

    Returns:
        numpy array shape (n_rows, n_features) — unscaled, matching training columns
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df = _parse_csv(filepath, artifacts_dir)
    elif ext in ('.pcap', '.pcapng'):
        df = _parse_pcap(filepath, artifacts_dir)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Upload a .csv or .pcap file.")

    return df.values.astype(np.float32)


# ── CSV parser ────────────────────────────────────────────────────────────────

def _parse_csv(filepath: str, artifacts_dir: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, low_memory=False)

    # Drop label columns if present (user might upload a labelled dataset)
    for col in ['Label', 'label', 'attack_cat', 'class']:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = _align_columns(df, artifacts_dir)
    return df


# ── PCAP parser ───────────────────────────────────────────────────────────────

def _parse_pcap(filepath: str, artifacts_dir: str) -> pd.DataFrame:
    """
    Extract network traffic features from a PCAP file using scapy.
    Produces the same feature columns as the UNSW-NB15 dataset.
    """
    try:
        from scapy.all import rdpcap, IP, TCP, UDP
    except ImportError:
        raise ImportError(
            "scapy is required to parse PCAP files.\n"
            "Install it with:  pip install scapy"
        )

    packets = rdpcap(filepath)
    records = []

    for pkt in packets:
        record = _extract_packet_features(pkt)
        if record:
            records.append(record)

    if not records:
        raise ValueError("No valid IP packets found in the PCAP file.")

    df = pd.DataFrame(records)
    df = _align_columns(df, artifacts_dir)
    return df


def _extract_packet_features(pkt) -> dict:
    """
    Extract per-packet features that approximate UNSW-NB15 columns.
    Extend this dict to match all columns your model was trained on.
    """
    from scapy.all import IP, TCP, UDP

    if not pkt.haslayer('IP'):
        return {}

    ip  = pkt['IP']
    rec = {
        'proto':    ip.proto,
        'ttl':      ip.ttl,
        'len':      ip.len,
        'id':       ip.id,
        'frag':     ip.frag,
        'tos':      ip.tos,
    }

    if pkt.haslayer('TCP'):
        tcp = pkt['TCP']
        rec.update({
            'sport':    tcp.sport,
            'dport':    tcp.dport,
            'seq':      tcp.seq % 1e6,        # mod to keep scale reasonable
            'ack':      tcp.ack % 1e6,
            'flags':    int(tcp.flags),
            'window':   tcp.window,
            'urgptr':   tcp.urgptr,
            'dataofs':  tcp.dataofs,
        })
    elif pkt.haslayer('UDP'):
        udp = pkt['UDP']
        rec.update({
            'sport':  udp.sport,
            'dport':  udp.dport,
            'ulen':   udp.len,
            # Fill TCP-specific fields with 0
            'seq': 0, 'ack': 0, 'flags': 0,
            'window': 0, 'urgptr': 0, 'dataofs': 0,
        })
    else:
        rec.update({
            'sport': 0, 'dport': 0, 'seq': 0, 'ack': 0,
            'flags': 0, 'window': 0, 'urgptr': 0, 'dataofs': 0,
        })

    # Payload length
    rec['payload_len'] = len(pkt.payload) if pkt.payload else 0

    return rec


# ── Column alignment ──────────────────────────────────────────────────────────

def _align_columns(df: pd.DataFrame, artifacts_dir: str) -> pd.DataFrame:
    """
    Ensure the parsed DataFrame has exactly the columns the model expects:
    - Load expected column list from saved artifacts
    - Drop extras, fill missing with 0
    - Encode any remaining categoricals
    """
    fnames_path = os.path.join(artifacts_dir, 'feature_names.npy')
    expected    = np.load(fnames_path, allow_pickle=True).tolist()

    # Encode object columns using saved encoders
    encoders_path = os.path.join(artifacts_dir, 'encoders.pkl')
    if os.path.exists(encoders_path):
        encoders = joblib.load(encoders_path)
        for col, enc in encoders.items():
            if col in df.columns:
                # Handle unseen labels gracefully
                df[col] = df[col].astype(str).map(
                    lambda v, e=enc: e.transform([v])[0]
                    if v in e.classes_ else 0
                )

    # Drop unexpected columns
    extra = [c for c in df.columns if c not in expected]
    df = df.drop(columns=extra)

    # Add missing columns as 0
    for col in expected:
        if col not in df.columns:
            df[col] = 0

    # Reorder to match training order
    df = df[expected]

    # Final numeric cast
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    return df