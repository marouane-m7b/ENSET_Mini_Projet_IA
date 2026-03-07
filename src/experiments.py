"""
experiments.py — Expériences comparatives et génération des figures
E.1 : UCS vs Greedy vs A* sur 3 grilles
E.2 : Variation de epsilon (impact Markov)
E.3 : h=0 vs Manhattan
E.4 : Weighted A*
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(__file__))
from astar import astar, ucs, greedy, weighted_astar, zero_heuristic, search, manhattan
from markov import build_policy, build_transition_matrix, compute_distribution, simulate_trajectories, absorption_analysis
from grids import GRIDS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Utilitaire : visualisation grille + chemin
# ─────────────────────────────────────────────
def plot_grid_path(grid, path, start, goal, title="", filename=None):
    rows, cols = len(grid), len(grid[0])
    img = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            img[r, c] = 1 if grid[r][c] == 1 else 0

    fig, ax = plt.subplots(figsize=(max(5, cols * 0.55), max(5, rows * 0.55)))
    cmap = ListedColormap(["#F0F0F0", "#333333"])
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1)

    if path:
        px = [p[1] for p in path]
        py = [p[0] for p in path]
        ax.plot(px, py, "b-", linewidth=2.5, zorder=3)

    ax.plot(start[1], start[0], "go", markersize=12, zorder=4, label="Départ")
    ax.plot(goal[1], goal[0], "r*", markersize=14, zorder=4, label="But")

    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.set_xticklabels(range(cols), fontsize=7)
    ax.set_yticklabels(range(rows), fontsize=7)
    ax.grid(True, color="gray", linewidth=0.3)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# E.1 : Comparaison UCS / Greedy / A*
# ─────────────────────────────────────────────
def experiment_1():
    print("=== E.1 : UCS vs Greedy vs A* ===")
    results = {}
    algos = {"UCS": ucs, "Greedy": greedy, "A*": astar}

    for gname, (grid, start, goal) in GRIDS.items():
        results[gname] = {}
        for aname, func in algos.items():
            r = func(grid, start, goal)
            results[gname][aname] = r
            status = "✓" if r["success"] else "✗"
            print(f"  {gname:6s} | {aname:6s} | {status} coût={r['cost']:.1f} "
                  f"nœuds={r['nodes_expanded']} OPEN_max={r['open_max_size']}")

        # Afficher chemin A*
        r_astar = results[gname]["A*"]
        if r_astar["success"]:
            plot_grid_path(grid, r_astar["path"], start, goal,
                           title=f"A* — grille {gname} (coût={r_astar['cost']:.1f})",
                           filename=os.path.join(OUTPUT_DIR, f"grid_{gname}_astar.png"))

    # Graphe comparatif
    grid_names = list(GRIDS.keys())
    algo_names = list(algos.keys())
    x = np.arange(len(grid_names))
    width = 0.25
    colors = ["#4E79A7", "#F28E2B", "#59A14F"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (ax, metric, label) in enumerate(zip(
        axes,
        ["nodes_expanded", "cost"],
        ["Nœuds développés", "Coût du chemin"],
    )):
        for j, (aname, col) in enumerate(zip(algo_names, colors)):
            vals = [results[g][aname][metric] for g in grid_names]
            ax.bar(x + j * width, vals, width, label=aname, color=col, alpha=0.85)
        ax.set_xlabel("Grille", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{label} par algorithme", fontsize=12, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(grid_names)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "e1_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  → figures sauvegardées\n")
    return results


# ─────────────────────────────────────────────
# E.2 : Variation de ε
# ─────────────────────────────────────────────
def experiment_2():
    print("=== E.2 : Impact de epsilon (Markov) ===")
    grid, start, goal = GRIDS["medium"]
    epsilons = [0.0, 0.1, 0.2, 0.3]

    r = astar(grid, start, goal)
    if not r["success"]:
        print("  A* échoué sur grille medium"); return {}

    path = r["path"]
    policy = build_policy(path)
    accessible = list({s for s in path})

    # Ajouter états voisins accessibles (pour les déviations)
    from astar import get_neighbors
    for s in path:
        for nb, _ in get_neighbors(s, grid):
            if nb not in accessible:
                accessible.append(nb)

    results = {}
    for eps in epsilons:
        P, s2i = build_transition_matrix(accessible, policy, goal, grid, epsilon=eps)
        goal_idx = s2i.get(goal, -1)
        n = P.shape[0]
        pi0 = np.zeros(n)
        start_idx = s2i.get(start)
        if start_idx is not None:
            pi0[start_idx] = 1.0

        probs_at_n = []
        steps_list = list(range(0, 100, 5)) + list(range(100, 201, 10))
        pi = pi0.copy()
        Pn = np.eye(n)
        prev_step = 0
        for step in steps_list:
            diff = step - prev_step
            for _ in range(diff):
                Pn = Pn @ P
            pi_n = pi0 @ Pn
            probs_at_n.append(pi_n[goal_idx] if goal_idx >= 0 else 0)
            prev_step = step

        # Simulation
        sim = simulate_trajectories(P, s2i, start, goal, n_traj=2000, max_steps=200)
        results[eps] = {
            "prob_goal_sim": sim["prob_goal"],
            "mean_time": sim["mean_time"],
            "prob_curve": probs_at_n,
            "steps": steps_list,
            "astar_cost": r["cost"],
            "reach_times": sim["reach_times"],
        }
        print(f"  ε={eps:.1f} | P(GOAL)_sim={sim['prob_goal']:.3f} | "
              f"Temps moyen={sim['mean_time']:.1f} | Coût A*={r['cost']:.1f}")

    # Figure : probabilité d'atteindre GOAL vs étapes
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_eps = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for eps, col in zip(epsilons, colors_eps):
        axes[0].plot(results[eps]["steps"], results[eps]["prob_curve"],
                     label=f"ε={eps}", color=col, linewidth=2)
    axes[0].set_xlabel("Étapes n", fontsize=11)
    axes[0].set_ylabel("P(X_n = GOAL)", fontsize=11)
    axes[0].set_title("Évolution π^(n)[GOAL] (calcul matriciel)", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10); axes[0].grid(alpha=0.4)

    eps_vals = list(results.keys())
    prob_sims = [results[e]["prob_goal_sim"] for e in eps_vals]
    mean_times = [results[e]["mean_time"] for e in eps_vals]
    axes[1].bar([str(e) for e in eps_vals], prob_sims, color=colors_eps[:len(eps_vals)], alpha=0.85)
    axes[1].set_xlabel("ε", fontsize=11)
    axes[1].set_ylabel("P(GOAL) — simulation", fontsize=11)
    axes[1].set_title("Probabilité d'atteindre GOAL (Monte-Carlo)", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 1.05); axes[1].grid(axis="y", alpha=0.4)
    for i, (v, mt) in enumerate(zip(prob_sims, mean_times)):
        axes[1].text(i, v + 0.02, f"{v:.2f}\n(t̄={mt:.0f})", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "e2_epsilon.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  → figures sauvegardées\n")
    return results


# ─────────────────────────────────────────────
# E.3 : h=0 vs Manhattan
# ─────────────────────────────────────────────
def experiment_3():
    print("=== E.3 : h=0 vs Manhattan ===")
    results = {}
    for gname, (grid, start, goal) in GRIDS.items():
        r0 = search(grid, start, goal, zero_heuristic, 1.0, 1.0)  # A* avec h=0 = UCS
        rm = astar(grid, start, goal)
        results[gname] = {"h=0": r0, "Manhattan": rm}
        print(f"  {gname:6s} | h=0: nœuds={r0['nodes_expanded']} coût={r0['cost']:.1f} "
              f"| Manhattan: nœuds={rm['nodes_expanded']} coût={rm['cost']:.1f}")

    gnames = list(GRIDS.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(gnames))
    ax.bar(x - 0.2, [results[g]["h=0"]["nodes_expanded"] for g in gnames],
           0.4, label="h = 0 (UCS)", color="#4E79A7", alpha=0.85)
    ax.bar(x + 0.2, [results[g]["Manhattan"]["nodes_expanded"] for g in gnames],
           0.4, label="h = Manhattan", color="#59A14F", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(gnames)
    ax.set_ylabel("Nœuds développés", fontsize=11)
    ax.set_title("Impact de l'heuristique sur l'exploration", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "e3_heuristic.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  → figures sauvegardées\n")
    return results


# ─────────────────────────────────────────────
# E.4 : Weighted A*
# ─────────────────────────────────────────────
def experiment_4():
    print("=== E.4 : Weighted A* ===")
    grid, start, goal = GRIDS["hard"]
    weights = [1.0, 1.5, 2.0, 3.0, 5.0]
    results = {}
    for w in weights:
        r = weighted_astar(grid, start, goal, w)
        results[w] = r
        ratio = r["cost"] / results[1.0]["cost"] if results[1.0]["cost"] > 0 else float("inf")
        print(f"  w={w:.1f} | nœuds={r['nodes_expanded']} coût={r['cost']:.1f} "
              f"ratio_coût={ratio:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    wvals = [str(w) for w in weights]
    axes[0].plot(wvals, [results[w]["nodes_expanded"] for w in weights],
                 "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Poids w", fontsize=11); axes[0].set_ylabel("Nœuds développés", fontsize=11)
    axes[0].set_title("Weighted A* — Nœuds développés", fontsize=12, fontweight="bold")
    axes[0].grid(alpha=0.4)

    opt_cost = results[1.0]["cost"]
    axes[1].plot(wvals, [results[w]["cost"] / opt_cost for w in weights],
                 "rs-", linewidth=2, markersize=8)
    axes[1].axhline(1.0, color="gray", linestyle="--", label="Optimal (w=1)")
    axes[1].set_xlabel("Poids w", fontsize=11)
    axes[1].set_ylabel("Ratio coût / coût optimal", fontsize=11)
    axes[1].set_title("Weighted A* — Sous-optimalité", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10); axes[1].grid(alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "e4_weighted.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  → figures sauvegardées\n")
    return results


# ─────────────────────────────────────────────
# Figure : distribution des temps d'atteinte
# ─────────────────────────────────────────────
def plot_reach_time_distribution(sim_results_by_eps):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    epsilons = [0.0, 0.1, 0.2, 0.3]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for ax, eps, col in zip(axes.flat, epsilons, colors):
        times = sim_results_by_eps.get(eps, {}).get("reach_times", [])
        if times:
            ax.hist(times, bins=30, color=col, alpha=0.8, edgecolor="white")
            ax.axvline(np.mean(times), color="black", linestyle="--",
                       label=f"Moyenne={np.mean(times):.1f}")
            ax.legend(fontsize=9)
        ax.set_title(f"ε = {eps}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Étapes pour atteindre GOAL", fontsize=9)
        ax.set_ylabel("Fréquence", fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle("Distribution du temps d'atteinte GOAL (2000 trajectoires)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "e2_reach_time_dist.png"), dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    r1 = experiment_1()
    r2 = experiment_2()
    experiment_3()
    experiment_4()
    plot_reach_time_distribution(r2)
    print(f"\nToutes les figures sont dans : {OUTPUT_DIR}")