"""Development-only map export; dashboard itself needs no networkx."""
from hashlib import sha256
import json
from pathlib import Path

from ai.topology import load_network

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    inp = ROOT / "data/Net1.inp"
    graph = load_network(inp)
    layout = {"source_sha256": sha256(inp.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "nodes": dict(graph.nodes(data=True)),
              "links": [d for _, _, d in graph.edges(data=True)]}
    (ROOT / "frontend/net1_layout.json").write_text(json.dumps(layout, indent=2) + "\n")
