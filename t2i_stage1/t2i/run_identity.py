"""Immutable run provenance, independent of scheduler IDs and target step counts."""
import hashlib
import json
from pathlib import Path
import uuid


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def directory_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    # Shared GPFS paths have different st_dev values on login/compute nodes.
    return {'path': str(path), 'inode': stat.st_ino, 'uid': stat.st_uid}


def new_run_identity(binding, starting_checkpoint=None):
    value = dict(schema='priming-stage1-run-v1', launch_id=uuid.uuid4().hex,
                 binding=json.loads(canonical_bytes(binding)), binding_sha256=digest(binding),
                 starting_checkpoint=starting_checkpoint)
    return dict(value, sha256=digest(value))


def validate_run_identity(value):
    try:
        payload = {key: value[key] for key in
                   ('schema', 'launch_id', 'binding', 'binding_sha256', 'starting_checkpoint')}
        valid = (set(value) == set(payload) | {'sha256'}
                 and payload['schema'] == 'priming-stage1-run-v1'
                 and uuid.UUID(hex=payload['launch_id']).hex == payload['launch_id']
                 and isinstance(payload['binding'], dict)
                 and payload['binding_sha256'] == digest(payload['binding'])
                 and value['sha256'] == digest(payload))
        parent = payload['starting_checkpoint']
        if parent is not None:
            valid = valid and (set(parent) == {'path', 'receipt_sha256', 'manifest_sha256',
                                             'run_identity_sha256', 'step'}
                and Path(parent['path']).is_absolute()
                and isinstance(parent['step'], int) and not isinstance(parent['step'], bool)
                and parent['step'] >= 0
                and all(isinstance(parent[k], str) and len(parent[k]) == 64
                        and all(c in '0123456789abcdef' for c in parent[k])
                        for k in ('receipt_sha256', 'manifest_sha256', 'run_identity_sha256')))
        if valid:
            return value
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    raise ValueError('Run identity is missing or invalid; use a new output directory')
