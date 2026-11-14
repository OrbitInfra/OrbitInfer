# Adapted from Resin a1989594 for OrbitInfer; changes: docs/source-import.md.
"""Wire fixtures for the private native distributed-HTTP packet boundary."""

import json

from orbitinfer._native import _HttpEvents, _HttpIngress


def ingress(requests):
    return _HttpIngress.from_bytes(json.dumps({"requests": requests}).encode())


def decoded_events(packet: _HttpEvents):
    return json.loads(packet.to_bytes() or b'{"events":[]}')["events"]
