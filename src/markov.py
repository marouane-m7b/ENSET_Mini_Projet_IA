"""
markov.py — Chaînes de Markov induites par une politique
"""
import numpy as np
from typing import List, Tuple, Dict

State = Tuple[int, int]

def build_policy(path: List[State]) -> Dict[State, State]:
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
) -> tuple:
    # Construction de la matrice P
    all_states = list(states)
    if goal not in all_states:
        all_states.append(goal)
    
    state_to_idx = {s: i for i, s in enumerate(all_states)}
    n = len(all_states)
    P = np.zeros((n, n))
    
    # TODO: Remplir P avec le modèle d'incertitude
    
    return P, state_to_idx
