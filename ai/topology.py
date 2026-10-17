"""Load the water network topology straight from an EPANET .inp file.

Pure-Python parser (no wntr), so it runs anywhere networkx does, including s390x.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

DEFAULT_INP_PATH = "data/Net1.inp"

NODE_SECTIONS = {"JUNCTIONS": "junction", "RESERVOIRS": "reservoir", "TANKS": "tank"}


def _read_sections(inp_path: str | Path) -> dict[str, list[list[str]]]:
    """Split an .inp file into {SECTION: [tokens per data line]}.

    Text after ';' is a comment; blank lines are skipped; parsing stops at [END].
    """
    sections: dict[str, list[list[str]]] = {}
    current: str | None = None
    with open(inp_path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("["):
                current = line.strip("[]").strip().upper()
                if current == "END":
                    break
                sections.setdefault(current, [])
                continue
            if current is not None:
                sections[current].append(line.split())
    return sections


def load_network(inp_path: str | Path = DEFAULT_INP_PATH) -> nx.Graph:
    """Build an undirected graph of the network described in an EPANET .inp file.

    Nodes carry: type (junction/reservoir/tank), x, y.
    Edges carry: pipe_id, kind (pipe/pump), length, diameter, and start/end,
    which keep the .inp direction (node1 -> node2) for flow sign conventions.
    """
    sections = _read_sections(inp_path)
    G = nx.Graph(source=str(inp_path))

    for section, node_type in NODE_SECTIONS.items():
        for tokens in sections.get(section, []):
            G.add_node(tokens[0], type=node_type, x=None, y=None)

    for node, x, y, *_ in sections.get("COORDINATES", []):
        if node in G:
            G.nodes[node]["x"] = float(x)
            G.nodes[node]["y"] = float(y)

    def add_link(link_id: str, start: str, end: str, kind: str,
                 length: float | None, diameter: float | None) -> None:
        missing = [n for n in (start, end) if n not in G]
        if missing:
            raise ValueError(f"{kind} {link_id} references unknown node(s) {missing} in {inp_path}")
        G.add_edge(start, end, pipe_id=link_id, kind=kind,
                   length=length, diameter=diameter, start=start, end=end)

    for link_id, start, end, length, diameter, *_ in sections.get("PIPES", []):
        add_link(link_id, start, end, "pipe", float(length), float(diameter))
    for link_id, start, end, *_ in sections.get("PUMPS", []):
        add_link(link_id, start, end, "pump", None, None)

    return G


def pipe_ends(G: nx.Graph, include_pumps: bool = True) -> dict[str, tuple[str, str]]:
    """Return {pipe_id: (start_node, end_node)} in .inp direction."""
    return {
        d["pipe_id"]: (d["start"], d["end"])
        for _, _, d in G.edges(data=True)
        if include_pumps or d["kind"] == "pipe"
    }


def junction_ids(G: nx.Graph) -> list[str]:
    """Return the ids of all junction nodes (excludes tanks and reservoirs)."""
    return [n for n, d in G.nodes(data=True) if d["type"] == "junction"]
