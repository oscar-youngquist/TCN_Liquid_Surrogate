"""Opt-in prepared-data audits; training and evaluation never call these helpers."""
from pathlib import Path
from .ported.storage import atomic_json, digest
import json


def fingerprint(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def verify_once(data, destination):
    data, destination = Path(data).resolve(), Path(destination)
    if destination.exists():
        check_receipt(data, destination)
        return
    manifest_hash = digest(data/'manifest.json')
    manifest = json.loads((data/'manifest.json').read_text())
    files = {}
    for name, expected in manifest['artifacts'].items():
        before = fingerprint(data/name)
        if digest(data/name) != expected or fingerprint(data/name) != before:
            raise ValueError(f'Prepared artifact changed or checksum mismatch: {name}')
        files[name] = before
    if digest(data/'manifest.json') != manifest_hash:
        raise ValueError('Manifest changed during verification')
    atomic_json(destination, dict(data=str(data), manifest_sha256=manifest_hash, files=files))


def check_receipt(data, receipt):
    data = Path(data).resolve()
    row = json.loads(Path(receipt).read_text())
    manifest = json.loads((data/'manifest.json').read_text())
    if (row['data'] != str(data) or row['manifest_sha256'] != digest(data/'manifest.json')
            or set(row['files']) != set(manifest['artifacts'])):
        raise ValueError('Verification receipt does not match dataset')
    for name, expected in row['files'].items():
        if fingerprint(data/name) != expected:
            raise ValueError(f'Prepared artifact changed since verification: {name}')
