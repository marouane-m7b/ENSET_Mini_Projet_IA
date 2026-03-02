"""
astar.py — Planification heuristique sur grille 2D
Implémente : UCS, Greedy Best-First, A*
"""
import heapq
from typing import List, Tuple, Dict, Optional, Callable

# Type alias
State = Tuple[int, int]


def manhattan(p: State, goal: State) -> float:
    return abs(p[0] - goal[0]) + abs(p[1] - goal[1])


def zero_heuristic(p: State, goal: State) -> float:
    return 0.0


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


def search(
    grid: List[List[int]],
    start: State,
    goal: State,
    heuristic: Callable = manhattan,
    weight_g: float = 1.0,
    weight_h: float = 1.0,
) -> Dict:
    """
    Algorithme de recherche générique.
      weight_g=1, weight_h=0  -> UCS
      weight_g=0, weight_h=1  -> Greedy
      weight_g=1, weight_h=1  -> A*
      weight_g=1, weight_h=w  -> Weighted A* (w>1)
    Retourne un dict avec : path, cost, nodes_expanded, open_max_size, success
    """
    OPEN = []          # heap de (f, g, state)
    CLOSED = set()
    g_scores: Dict[State, float] = {start: 0.0}
    came_from: Dict[State, Optional[State]] = {start: None}
    nodes_expanded = 0
    open_sizes = []

    h0 = heuristic(start, goal)
    f0 = weight_g * 0.0 + weight_h * h0
    heapq.heappush(OPEN, (f0, 0.0, start))

    while OPEN:
        open_sizes.append(len(OPEN))
        f, g, current = heapq.heappop(OPEN)

        if current in CLOSED:
            continue
        CLOSED.add(current)
        nodes_expanded += 1

        if current == goal:
            # Reconstruire le chemin
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return {
                "success": True,
                "path": path,
                "cost": g_scores[goal],
                "nodes_expanded": nodes_expanded,
                "open_max_size": max(open_sizes) if open_sizes else 0,
            }

        for neighbor, step_cost in get_neighbors(current, grid):
            if neighbor in CLOSED:
                continue
            tentative_g = g + step_cost
            if tentative_g < g_scores.get(neighbor, float("inf")):
                g_scores[neighbor] = tentative_g
                came_from[neighbor] = current
                h = heuristic(neighbor, goal)
                f = weight_g * tentative_g + weight_h * h
                heapq.heappush(OPEN, (f, tentative_g, neighbor))

    return {"success": False, "path": [], "cost": float("inf"),
            "nodes_expanded": nodes_expanded, "open_max_size": max(open_sizes) if open_sizes else 0}


def astar(grid, start, goal):
    return search(grid, start, goal, manhattan, 1.0, 1.0)

def ucs(grid, start, goal):
    return search(grid, start, goal, zero_heuristic, 1.0, 0.0)

def greedy(grid, start, goal):
    return search(grid, start, goal, manhattan, 0.0, 1.0)

def weighted_astar(grid, start, goal, w=2.0):
    return search(grid, start, goal, manhattan, 1.0, w)
