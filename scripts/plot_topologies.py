"""
Plot all 4 power grid topologies as clean network graphs.
Output: artifacts/topologies_images/{name}.png  +  all_topologies.png

Run: python3 scripts/plot_topologies.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandapower as pp
import pandapower.networks as pn
import numpy as np

OUT_DIR = "artifacts/topologies_images"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Network definitions ──────────────────────────────────────────
NETWORKS = [
    ("IEEE 13-bus",  "13-bus",  pn.case14(),   "#2196F3"),   # blue
    ("IEEE 34-bus",  "34-bus",  pn.case33bw(), "#4CAF50"),   # green
    ("IEEE 57-bus",  "57-bus",  pn.case57(),   "#FF9800"),   # orange
    ("IEEE 123-bus", "123-bus", pn.case118(),  "#E91E63"),   # pink
]


def build_graph(net):
    """Build NetworkX graph from pandapower network."""
    G = nx.Graph()
    for _, row in net.bus.iterrows():
        G.add_node(row.name, vn=row.vn_kv)
    for _, row in net.line.iterrows():
        G.add_edge(row.from_bus, row.to_bus,
                   length=row.length_km if 'length_km' in row else 1.0,
                   etype='line')
    if hasattr(net, 'trafo') and len(net.trafo) > 0:
        for _, row in net.trafo.iterrows():
            G.add_edge(row.hv_bus, row.lv_bus, etype='trafo')
    return G


def node_colors(net, G):
    """Colour nodes: red=slack/generator, blue=load, gray=transit."""
    colors = {}
    gen_buses  = set(net.gen.bus.tolist())  if len(net.gen)  > 0 else set()
    ext_buses  = set(net.ext_grid.bus.tolist()) if len(net.ext_grid) > 0 else set()
    load_buses = set(net.load.bus.tolist()) if len(net.load) > 0 else set()
    for n in G.nodes():
        if n in ext_buses:
            colors[n] = '#e53935'       # slack — red
        elif n in gen_buses:
            colors[n] = '#fb8c00'       # generator — orange
        elif n in load_buses:
            colors[n] = '#1e88e5'       # load bus — blue
        else:
            colors[n] = '#90a4ae'       # transit — gray
    return [colors[n] for n in G.nodes()]


def smart_layout(G, net, seed=42):
    """Use real bus geodata coordinates if available, else spring layout."""
    if len(net.bus_geodata) > 0:
        pos = {}
        for bus_id in G.nodes():
            if bus_id in net.bus_geodata.index:
                x = net.bus_geodata.at[bus_id, 'x']
                y = net.bus_geodata.at[bus_id, 'y']
                pos[bus_id] = (x, y)
        if len(pos) == len(G.nodes()):
            return pos
    n = len(G.nodes())
    if n <= 40:
        try:
            return nx.kamada_kawai_layout(G)
        except Exception:
            pass
    return nx.spring_layout(G, seed=seed, k=2.5/np.sqrt(n), iterations=100)


def plot_single(title, fname, net, color, ax=None, standalone=True):
    G   = build_graph(net)
    pos = smart_layout(G, net)
    nc  = node_colors(net, G)
    n_buses = len(net.bus)
    n_lines = len(net.line) + (len(net.trafo) if hasattr(net, 'trafo') else 0)

    node_size = max(20, min(180, 2800 // n_buses))
    font_size = max(4, min(8, 80 // n_buses))

    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        ax.set_facecolor('white')

    # Edge colours
    edge_colors = []
    for u, v, d in G.edges(data=True):
        edge_colors.append('#455a64' if d.get('etype') == 'trafo' else '#b0bec5')

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           width=1.2, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc,
                           node_size=node_size, alpha=0.92)

    if n_buses <= 30:
        nx.draw_networkx_labels(G, pos, ax=ax,
                                font_size=font_size, font_color='white',
                                font_weight='bold')

    ax.set_title(f"{title}\n{n_buses} buses · {n_lines} branches",
                 fontsize=13, fontweight='bold', color=color, pad=10)
    ax.axis('off')

    # Legend
    legend_items = [
        mpatches.Patch(color='#e53935', label='Slack / External Grid'),
        mpatches.Patch(color='#fb8c00', label='Generator Bus'),
        mpatches.Patch(color='#1e88e5', label='Load Bus'),
        mpatches.Patch(color='#90a4ae', label='Transit Bus'),
    ]
    ax.legend(handles=legend_items, loc='lower right',
              fontsize=8, framealpha=0.9)

    if standalone:
        plt.tight_layout()
        out = f"{OUT_DIR}/{fname}.png"
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {out}")
        return out


def plot_all_grid():
    """2×2 grid of all 4 topologies."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    fig.suptitle(
        "Power Grid Topologies — IEEE Test Cases\n"
        "QE-SAC-FL: Federated Quantum RL across Heterogeneous Clients",
        fontsize=14, fontweight='bold', y=1.01
    )

    for ax, (title, fname, net, color) in zip(axes.flat, NETWORKS):
        ax.set_facecolor('white')
        plot_single(title, fname, net, color, ax=ax, standalone=False)

    plt.tight_layout()
    out = f"{OUT_DIR}/all_topologies.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")
    return out


def main():
    print(f"Output folder: {OUT_DIR}\n")
    print("Plotting individual topologies...")
    for title, fname, net, color in NETWORKS:
        n_buses = len(net.bus)
        n_lines = len(net.line)
        print(f"  {title}: {n_buses} buses, {n_lines} lines")
        plot_single(title, fname, net, color)

    print("\nPlotting 2×2 grid...")
    plot_all_grid()
    print("\nDone.")


if __name__ == "__main__":
    main()
