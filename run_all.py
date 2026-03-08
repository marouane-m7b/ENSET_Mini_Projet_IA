#!/usr/bin/env python3
"""
run_all.py — Script principal pour exécuter toutes les expériences
"""
import sys
import os

# Ajouter src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("=" * 70)
    print("MINI-PROJET : Planification Robuste sur Grille (A* + Markov)")
    print("=" * 70)
    print()
    
    # Expériences principales
    print("1️⃣  Exécution des expériences E.1-E.4...")
    print("-" * 70)
    from experiments import experiment_1, experiment_2, experiment_3, experiment_4, plot_reach_time_distribution
    
    r1 = experiment_1()
    r2 = experiment_2()
    r3 = experiment_3()
    r4 = experiment_4()
    plot_reach_time_distribution(r2)
    
    print()
    print("2️⃣  Génération des figures de trajectoires...")
    print("-" * 70)
    from gen_traj_figures import figure_medium_epsilons, figure_3grids_heatmap
    
    figure_medium_epsilons()
    figure_3grids_heatmap()
    
    print()
    print("3️⃣  Génération des figures supplémentaires...")
    print("-" * 70)
    from gen_extra_figures import figure_aperiodicity, compute_absorption_table
    
    figure_aperiodicity()
    print()
    print("Tableau absorption (analytique vs Monte-Carlo):")
    compute_absorption_table()
    
    print()
    print("=" * 70)
    print("✅ TERMINÉ ! Toutes les figures sont dans le dossier 'figures/'")
    print("=" * 70)

if __name__ == "__main__":
    main()
