"""
gen_extra_figures.py — Figures supplémentaires pour le rapport:
  - aperiodicite.png  : preuve visuelle apériodicité (self-loops)
  - absorption_table  : données pour tableau analytique vs MC
  - networkx_classes  : graphe classes de communication (grille facile)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from astar import astar, get_neighbors
from markov import build_policy, build_transition_matrix, simulate_trajectories
from grids import GRIDS

FIGS = os.path.join(os.path.dirname(__file__), 'figures')

# ── Figure : Apériodicité (self-loops visuels) ────────────────────────────────
def figure_aperiodicity():
    grid, start, goal = GRIDS['medium']
    r = astar(grid, start, goal); path = r['path']
    pol = build_policy(path)
    acc = list({s for s in path})
    for s in path:
        for nb, _ in get_neighbors(s, grid):
            if nb not in acc: acc.append(nb)
    P, s2i = build_transition_matrix(acc, pol, goal, grid, epsilon=0.2)

    # Find states with self-loop
    self_loop_states = [(s, P[s2i[s], s2i[s]])
                        for s in acc if P[s2i[s], s2i[s]] > 1e-9]

    rows, cols = len(grid), len(grid[0])
    img = np.array([[1 if grid[r][c]==1 else 0
                     for c in range(cols)] for r in range(rows)])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap=ListedColormap(['#F2F2F2','#2D2D2D']), vmin=0, vmax=1, zorder=1)
    ax.grid(True, color='gray', lw=0.3, zorder=2)

    # Plan A*
    ax.plot([p[1] for p in path], [p[0] for p in path],
            'b--', lw=2, zorder=4, alpha=0.6, label='Plan A*')

    # Self-loop states as circles with size proportional to p_ii
    for s, pii in self_loop_states:
        size = pii * 800 + 100
        color = '#FF6600' if s != goal else '#1A6B2A'
        ax.scatter(s[1], s[0], s=size, c=color, alpha=0.7, zorder=5,
                   edgecolors='black', lw=0.5)

    ax.plot(start[1], start[0], 'go', ms=12, zorder=7)
    ax.plot(goal[1], goal[0], 'r*', ms=15, zorder=7)
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.set_xticklabels(range(cols), fontsize=8)
    ax.set_yticklabels(range(rows), fontsize=8)

    legend_els = [
        Line2D([0],[0], color='b', lw=2, ls='--', label='Plan A* (optimal)'),
        plt.scatter([],[], s=200, c='#FF6600', alpha=0.7,
                    edgecolors='black', label=f'Boucle p_ii>0 ({len(self_loop_states)} etats)'),
    ]
    ax.legend(handles=[
        Line2D([0],[0], color='b', lw=2, ls='--', label='Plan A*'),
        Line2D([0],[0], marker='o', color='#FF6600', ms=10, lw=0,
               label=f'Etat avec boucle p_ii>0 ({len(self_loop_states)} sur {len(acc)})'),
    ], fontsize=9, loc='upper right')
    ax.set_title('Preuve d\'aperiodicite : etats avec boucle propre ($p_{ii}>0$)\n'
                 'Grille medium, $\\varepsilon=0.2$ — taille cercle proportionnelle a $p_{ii}$',
                 fontweight='bold', fontsize=11)
    plt.tight_layout()
    out = f'{FIGS}/aperiodicity.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  -> {out}  ({len(self_loop_states)} self-loop states)')
    return len(self_loop_states)


# ── Tableau comparatif analytique vs MC ──────────────────────────────────────
def compute_absorption_table():
    grid, start, goal = GRIDS['medium']
    r = astar(grid, start, goal); path = r['path']
    pol = build_policy(path)
    acc = list({s for s in path})
    for s in path:
        for nb, _ in get_neighbors(s, grid):
            if nb not in acc: acc.append(nb)

    rows_data = []
    for eps in [0.0, 0.1, 0.2, 0.3]:
        P, s2i = build_transition_matrix(acc, pol, goal, grid, epsilon=eps)
        n = P.shape[0]
        goal_idx  = s2i[goal]
        start_idx = s2i[start]
        Pn = np.linalg.matrix_power(P, 200)
        p_matrix = float(Pn[start_idx, goal_idx])
        sim = simulate_trajectories(P, s2i, start, goal, n_traj=2000, max_steps=200)
        rows_data.append({
            'eps': eps,
            'p_matrix': p_matrix,
            'p_mc': sim['prob_goal'],
            'mean_time': sim['mean_time'],
            'ecart_pct': abs(p_matrix - sim['prob_goal']) * 100,
        })
        print(f"  eps={eps}: P^n={p_matrix:.4f}, MC={sim['prob_goal']:.4f}, "
              f"t_moy={sim['mean_time']:.1f}, ecart={abs(p_matrix-sim['prob_goal'])*100:.2f}%")
    return rows_data


if __name__ == '__main__':
    print('=== Figures supplementaires ===')
    figure_aperiodicity()
    print('\n=== Tableau absorption (analytique vs MC) ===')
    compute_absorption_table()
    print('Termine.')
