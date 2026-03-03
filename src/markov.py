"""
markov.py — Chaînes de Markov induites par une politique
"""
import numpy as np
from typing import List, Tuple, Dict

State = Tuple[int, int]

def build_policy(path: List[State]) -> Dict[State, State]:
    """Construit la politique déterministe à partir du chemin A*."""
    policy = {}
    for i in range(len(path) - 1):
        policy[path[i]] = path[i + 1]
    return policy

# TODO: build_transition_matrix()
# TODO: simulate_trajectories()
