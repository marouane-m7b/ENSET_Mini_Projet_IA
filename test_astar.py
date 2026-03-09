import sys
sys.path.insert(0, 'src')
from astar import astar
from grids import GRIDS

print("=== Tests de validation ===")
for name, (grid, start, goal) in GRIDS.items():
    r = astar(grid, start, goal)
    assert r['success'], f"A* failed on {name}"
    assert r['cost'] > 0, f"Invalid cost on {name}"
    print(f"✓ {name}: cost={r['cost']}, nodes={r['nodes_expanded']}")

print("\nTous les tests passent!")
