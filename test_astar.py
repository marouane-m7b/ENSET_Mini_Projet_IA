import sys
sys.path.insert(0, 'src')
from astar import astar
from grids import GRIDS

for name, (grid, start, goal) in GRIDS.items():
    r = astar(grid, start, goal)
    print(f"{name}: cost={r['cost']}, nodes={r['nodes_expanded']}")
