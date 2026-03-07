"""
gen_traj_figures.py — Génère les figures de trajectoires stochastiques
Figures produites :
  - traj_medium_epsilons.png  : grille medium, 3 valeurs de epsilon
  - traj_3grids_heatmap.png   : 3 grilles x 2 lignes (trajectoires + heatmap)
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
from markov import build_policy, build_transition_matrix
from grids import GRIDS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def simulate_full_trajectories(P, s2i, start, goal,
                                n_traj=50, max_steps=200, seed=42):
    """
    Simule n_traj trajectoires complètes et retourne deux listes :
      succ : trajectoires ayant atteint GOAL
      fail : trajectoires ayant échoué (dépassé max_steps)
    Chaque trajectoire est une liste de positions (tuple).
    """
    idx_to_state = {v: k for k, v in s2i.items()}
    start_idx = s2i.get(start)
    goal_idx  = s2i.get(goal)
    n = P.shape[0]
    P_cum = np.cumsum(P, axis=1)
    rng = np.random.default_rng(seed)
    succ, fail = [], []
    for _ in range(n_traj):
        state = start_idx
        traj  = [idx_to_state[state]]
        reached = False
        for _ in range(max_steps):
            nxt = min(int(np.searchsorted(P_cum[state], rng.random())), n - 1)
            state = nxt
            s = idx_to_state.get(state)
            if s:
                traj.append(s)
            if state == goal_idx:
                reached = True
                break
        (succ if reached else fail).append(traj)
    return succ, fail


def _build_P_for_grid(grid, start, goal, eps):
    """Construit P + politique pour une grille donnée."""
    r_a   = astar(grid, start, goal)
    path  = r_a['path']
    pol   = build_policy(path)
    acc   = list({s for s in path})
    for s in path:
        for nb, _ in get_neighbors(s, grid):
            if nb not in acc:
                acc.append(nb)
    P, s2i = build_transition_matrix(acc, pol, goal, grid, epsilon=eps)
    return P, s2i, path


def _render_grid(ax, grid):
    rows, cols = len(grid), len(grid[0])
    img = np.array([[1 if grid[r][c] == 1 else 0
                     for c in range(cols)] for r in range(rows)])
    ax.imshow(img, cmap=ListedColormap(['#F2F2F2', '#2D2D2D']),
              vmin=0, vmax=1, zorder=1)
    ax.grid(True, color='gray', lw=0.2, zorder=2)
    return rows, cols


def _draw_trajectories(ax, succ, fail, path, start, goal,
                        small=False):
    ms = 9 if small else 11
    for traj in fail:
        ax.plot([p[1] for p in traj], [p[0] for p in traj],
                '#FF3333', alpha=0.45, lw=1.2, zorder=3)
    for traj in succ:
        ax.plot([p[1] for p in traj], [p[0] for p in traj],
                '#00BB00', alpha=0.55, lw=1.2, zorder=4)
    ax.plot([p[1] for p in path], [p[0] for p in path],
            color='#0055FF', lw=2.2, ls='--', zorder=5)
    ax.plot(start[1], start[0], 'go', ms=ms, zorder=7)
    ax.plot(goal[1],  goal[0],  'r*', ms=ms+2, zorder=7)


LEGEND_ELEMENTS = [
    Line2D([0],[0], color='#0055FF', lw=2, ls='--',
           label='Plan A* (optimal)'),
    Line2D([0],[0], color='#00BB00', lw=2, alpha=0.9,
           label='Trajectoire reussie (GOAL atteint)'),
    Line2D([0],[0], color='#FF3333', lw=2, alpha=0.9,
           label='Trajectoire echouee (perdue)'),
]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 : grille medium, 3 valeurs de epsilon
# ─────────────────────────────────────────────────────────────────────────────
def figure_medium_epsilons(n_traj=50, epsilons=(0.1, 0.2, 0.3), seed=42):
    grid, start, goal = GRIDS['medium']
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    for ax, eps in zip(axes, epsilons):
        P, s2i, path = _build_P_for_grid(grid, start, goal, eps)
        succ, fail   = simulate_full_trajectories(P, s2i, start, goal,
                                                  n_traj=n_traj, seed=seed)
        rows, cols = _render_grid(ax, grid)
        _draw_trajectories(ax, succ, fail, path, start, goal)
        pct = len(succ) / n_traj * 100
        ax.set_title(f'$\\varepsilon={eps}$ — '
                     f'{len(succ)}/{n_traj} succes ({pct:.0f}%)',
                     fontweight='bold', fontsize=11)
        ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
        ax.set_xticklabels(range(cols), fontsize=7)
        ax.set_yticklabels(range(rows), fontsize=7)

    fig.legend(handles=LEGEND_ELEMENTS, loc='lower center', ncol=3,
               fontsize=10, framealpha=0.95, bbox_to_anchor=(0.5, -0.04))
    plt.suptitle(
        f'Trajectoires stochastiques — Grille medium, '
        f'{n_traj} trajectoires par $\\varepsilon$',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'traj_medium_epsilons.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  -> {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 : 3 grilles x 2 lignes (trajectoires + heatmap)
# ─────────────────────────────────────────────────────────────────────────────
def figure_3grids_heatmap(eps=0.2, n_traj=40, seed=42):
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))

    for col, (gname, (grid, start, goal)) in enumerate(GRIDS.items()):
        P, s2i, path = _build_P_for_grid(grid, start, goal, eps)
        succ, fail   = simulate_full_trajectories(P, s2i, start, goal,
                                                  n_traj=n_traj, seed=seed)
        rows, cols2 = len(grid), len(grid[0])
        mask = np.array([[grid[r][c] == 1
                          for c in range(cols2)] for r in range(rows)])

        # ── Ligne 0 : trajectoires individuelles ─────────────────────────────
        ax = axes[0][col]
        _render_grid(ax, grid)
        _draw_trajectories(ax, succ, fail, path, start, goal, small=True)
        pct = len(succ) / n_traj * 100
        ax.set_title(f'Grille {gname} — '
                     f'{len(succ)}/{n_traj} succes ({pct:.0f}%)',
                     fontweight='bold', fontsize=10)
        ax.set_xticks(range(cols2)); ax.set_yticks(range(rows))
        ax.set_xticklabels(range(cols2), fontsize=6)
        ax.set_yticklabels(range(rows), fontsize=6)

        # ── Ligne 1 : heatmap densité ─────────────────────────────────────────
        hm_s = np.zeros((rows, cols2))
        hm_f = np.zeros((rows, cols2))
        for traj in succ:
            for (r, c) in traj: hm_s[r, c] += 1
        for traj in fail:
            for (r, c) in traj: hm_f[r, c] += 1

        ax2 = axes[1][col]
        _render_grid(ax2, grid)
        hm_f_m = np.ma.masked_where(mask | (hm_f == 0), hm_f)
        ax2.imshow(hm_f_m, cmap='Reds', alpha=0.85, zorder=2,
                   vmin=0, vmax=max(hm_f.max(), 1))
        hm_s_m = np.ma.masked_where(mask | (hm_s == 0), hm_s)
        ax2.imshow(hm_s_m, cmap='Greens', alpha=0.85, zorder=3,
                   vmin=0, vmax=max(hm_s.max(), 1))
        ax2.plot([p[1] for p in path], [p[0] for p in path],
                 color='#0055FF', lw=2.2, ls='--', zorder=6)
        ax2.plot(start[1], start[0], 'go', ms=9,  zorder=7)
        ax2.plot(goal[1],  goal[0],  'r*', ms=11, zorder=7)
        ax2.set_title(f'Densite de passage — {gname}\n'
                      f'(vert=succes / rouge=echecs)',
                      fontweight='bold', fontsize=10)
        ax2.set_xticks(range(cols2)); ax2.set_yticks(range(rows))
        ax2.set_xticklabels(range(cols2), fontsize=6)
        ax2.set_yticklabels(range(rows), fontsize=6)
        ax2.grid(True, color='gray', lw=0.15, zorder=4)

    fig.legend(handles=LEGEND_ELEMENTS, loc='lower center', ncol=3,
               fontsize=10, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(
        f'Trajectoires stochastiques sur 3 grilles '
        f'($\\varepsilon={eps}$, {n_traj} chacune)\n'
        f'Ligne 1 : chemins individuels    '
        f'Ligne 2 : densite de passage cumulee',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'traj_3grids_heatmap.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  -> {out}')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Génération des figures de trajectoires ===')
    figure_medium_epsilons()
    figure_3grids_heatmap()
    print('Terminé.')
