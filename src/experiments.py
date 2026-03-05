import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from astar import astar, ucs, greedy
from grids import GRIDS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def experiment_1():
    print("=== E.1 : UCS vs Greedy vs A* ===")
    results = {}
    algos = {"UCS": ucs, "Greedy": greedy, "A*": astar}
    
    for gname, (grid, start, goal) in GRIDS.items():
        results[gname] = {}
        for aname, func in algos.items():
            r = func(grid, start, goal)
            results[gname][aname] = r
            print(f"  {gname} | {aname} | cost={r['cost']:.1f} nodes={r['nodes_expanded']}")
    
    # Graphique comparatif
    grid_names = list(GRIDS.keys())
    algo_names = list(algos.keys())
    x = np.arange(len(grid_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for j, aname in enumerate(algo_names):
        vals = [results[g][aname]["nodes_expanded"] for g in grid_names]
        ax.bar(x + j * width, vals, width, label=aname)
    
    ax.set_xlabel("Grille")
    ax.set_ylabel("Nœuds développés")
    ax.set_title("Comparaison des algorithmes")
    ax.set_xticks(x + width)
    ax.set_xticklabels(grid_names)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "e1_comparison.png"), dpi=120)
    plt.close()
    print("  → figure sauvegardée")
    
    return results

if __name__ == "__main__":
    experiment_1()
