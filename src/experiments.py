import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from astar import astar, ucs, greedy
from grids import GRIDS

def experiment_1():
    print("=== E.1 : UCS vs Greedy vs A* ===")
    results = {}
    algos = {"UCS": ucs, "Greedy": greedy, "A*": astar}
    
    for gname, (grid, start, goal) in GRIDS.items():
        results[gname] = {}
        for aname, func in algos.items():
            r = func(grid, start, goal)
            results[gname][aname] = r
            print(f"  {gname} | {aname} | cost={r['cost']:.1f} nodes={r['nodes_expanded']}")
    
    return results

if __name__ == "__main__":
    experiment_1()
