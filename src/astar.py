"""
astar.py — Planification heuristique sur grille 2D
"""
import heapq
from typing import List, Tuple, Dict

State = Tuple[int, int]

def manhattan(p: State, goal: State) -> float:
    return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

def get_neighbors(state: State, grid: List[List[int]]) -> List[Tuple[State, float]]:
    """Retourne les voisins libres (4-connexité) avec leurs coûts."""
    rows, cols = len(grid), len(grid[0])
    x, y = state
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 1:
            cost = float(grid[nx][ny]) if grid[nx][ny] > 1 else 1.0
            neighbors.append(((nx, ny), cost))
    return neighbors

# TODO: Implémenter search()
