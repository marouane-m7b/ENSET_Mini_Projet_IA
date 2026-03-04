"""
markov.py — Chaînes de Markov induites par une politique sur grille 2D
"""
import numpy as np
from typing import List, Tuple, Dict, Optional

State = Tuple[int, int]
GOAL_IDX = -1  # état absorbant GOAL
FAIL_IDX = -2  # état absorbant FAIL (si utilisé)


def build_policy(path: List[State]) -> Dict[State, State]:
    """Construit la politique déterministe à partir du chemin A*."""
    policy = {}
    for i in range(len(path) - 1):
        policy[path[i]] = path[i + 1]
    return policy


def build_transition_matrix(
    states: List[State],
    policy: Dict[State, State],
    goal: State,
    grid: List[List[int]],
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, Dict[State, int]]:
    """
    Construit la matrice de transition P (stochastique) sur les états accessibles.
    Les états sont indexés de 0..N-1. Le GOAL est l'état N.
    
    Modèle d'incertitude :
      - action voulue avec probabilité 1 - epsilon
      - déviation latérale gauche/droite avec probabilité epsilon/2 chacune
      - si déviation bloquée (obstacle/bord), reste sur place
    """
    # Construire index : états + GOAL
    all_states = list(states)
    if goal not in all_states:
        all_states.append(goal)

    # Ajouter état FAIL symbolique (dernier index)
    state_to_idx = {s: i for i, s in enumerate(all_states)}
    goal_idx = state_to_idx[goal]
    n = len(all_states)

    P = np.zeros((n, n))

    def try_move(pos, action):
        """Retourne la nouvelle position ou pos si bloqué."""
        rows, cols = len(grid), len(grid[0])
        nx, ny = pos[0] + action[0], pos[1] + action[1]
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 1:
            return (nx, ny)
        return pos  # reste sur place

    def lateral(action):
        """Retourne les deux déviations latérales d'une action (dx,dy)."""
        dx, dy = action
        return [(-dy, dx), (dy, -dx)]

    for s in states:
        idx = state_to_idx.get(s)
        if idx is None:
            continue

        # État absorbant GOAL
        if s == goal:
            P[idx, idx] = 1.0
            continue

        # Pas de politique pour cet état : reste sur place
        if s not in policy:
            P[idx, idx] = 1.0
            continue

        intended_next = policy[s]
        action = (intended_next[0] - s[0], intended_next[1] - s[1])
        lats = lateral(action)

        # Action voulue
        actual_next = try_move(s, action)
        prob_main = 1.0 - epsilon

        # Déviations
        lat_nexts = [try_move(s, lat) for lat in lats]
        prob_lat = epsilon / 2.0

        # Ajouter probabilités
        def add_prob(dest, prob):
            d_idx = state_to_idx.get(dest)
            if d_idx is not None:
                P[idx, d_idx] += prob
            else:
                # destination hors grille connue : reste sur place
                P[idx, idx] += prob

        add_prob(actual_next, prob_main)
        for ln in lat_nexts:
            add_prob(ln, prob_lat)

    # Vérification stochasticité
    row_sums = P.sum(axis=1)
    for i in range(n):
        if abs(row_sums[i] - 1.0) > 1e-9:
            if row_sums[i] > 0:
                P[i] /= row_sums[i]
            else:
                P[i, i] = 1.0  # état absorbant de secours

    return P, state_to_idx


def compute_distribution(P: np.ndarray, pi0: np.ndarray, n_steps: int) -> np.ndarray:
    """Calcule π^(n) = π^(0) * P^n."""
    pi = pi0.copy()
    for _ in range(n_steps):
        pi = pi @ P
    return pi


def absorption_analysis(P: np.ndarray, state_to_idx: Dict, transient_states: List, absorbing_states: List):
    """
    Calcule les probabilités d'absorption et le temps moyen.
    Décomposition P = [[I, 0], [R, Q]]
    N = (I - Q)^{-1}  (matrice fondamentale)
    B = N * R          (probabilités d'absorption)
    t = N * 1          (temps moyen avant absorption)
    """
    idx_trans = [state_to_idx[s] for s in transient_states if s in state_to_idx]
    idx_abs = [state_to_idx[s] for s in absorbing_states if s in state_to_idx]

    if not idx_trans or not idx_abs:
        return None

    Q = P[np.ix_(idx_trans, idx_trans)]
    R = P[np.ix_(idx_trans, idx_abs)]

    I = np.eye(len(idx_trans))
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        return None

    B = N @ R          # prob d'absorption
    t = N @ np.ones(len(idx_trans))  # temps moyen

    return {"N": N, "B": B, "t": t,
            "transient_states": transient_states,
            "absorbing_states": absorbing_states}


def simulate_trajectories(
    P: np.ndarray,
    state_to_idx: Dict,
    start: State,
    goal: State,
    n_traj: int = 1000,
    max_steps: int = 200,
) -> Dict:
    """
    Simulation Monte-Carlo de N trajectoires Markov.
    Retourne : prob_goal, dist_temps, taux_echec
    """
    idx_to_state = {v: k for k, v in state_to_idx.items()}
    start_idx = state_to_idx.get(start)
    goal_idx = state_to_idx.get(goal)
    n = P.shape[0]

    if start_idx is None or goal_idx is None:
        return {"prob_goal": 0, "mean_time": 0, "fail_rate": 1}

    # Cumulative P pour sampling efficace
    P_cum = np.cumsum(P, axis=1)

    reach_times = []
    failures = 0

    rng = np.random.default_rng(42)

    for _ in range(n_traj):
        state = start_idx
        reached = False
        for step in range(1, max_steps + 1):
            r = rng.random()
            next_state = np.searchsorted(P_cum[state], r)
            next_state = min(next_state, n - 1)
            state = next_state
            if state == goal_idx:
                reach_times.append(step)
                reached = True
                break
        if not reached:
            failures += 1

    prob_goal = len(reach_times) / n_traj
    mean_time = float(np.mean(reach_times)) if reach_times else float("inf")
    fail_rate = failures / n_traj

    return {
        "prob_goal": prob_goal,
        "mean_time": mean_time,
        "fail_rate": fail_rate,
        "reach_times": reach_times,
    }
