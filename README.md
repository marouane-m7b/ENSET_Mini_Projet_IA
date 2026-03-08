# Planification Robuste sur Grille : A* et Chaînes de Markov

## Contexte

Ce projet implémente une approche hybride combinant recherche heuristique et modélisation stochastique pour la planification de chemins robustes dans un environnement incertain. L'agent doit naviguer sur une grille 2D avec obstacles en minimisant le coût tout en tenant compte de l'incertitude des transitions.

## Approche

### Recherche Heuristique (A*)
- Planification de chemin optimal via l'algorithme A* avec f(n) = g(n) + h(n)
- Heuristique Manhattan admissible garantissant l'optimalité
- Structures de données efficaces (heap pour OPEN, set pour CLOSED)
- Variantes implémentées : UCS, Greedy Best-First, Weighted A*

### Modélisation Stochastique (Chaînes de Markov)
- Matrice de transition P modélisant l'incertitude des actions
- Paramètre ε contrôlant le taux d'erreur (déviations latérales)
- Analyse théorique : classes de communication, états absorbants, apériodicité
- Validation empirique par simulation Monte-Carlo

## Structure du Projet

```
.
├── src/
│   ├── astar.py              # Algorithmes de recherche heuristique
│   ├── markov.py             # Chaînes de Markov et simulation
│   ├── grids.py              # Définition des environnements de test
│   ├── experiments.py        # Expériences comparatives (E.1-E.4)
│   ├── gen_traj_figures.py   # Visualisation des trajectoires stochastiques
│   └── gen_extra_figures.py  # Analyses supplémentaires
├── figures/                  # Résultats graphiques
├── run_all.py               # Script d'exécution principal
├── notebook_mini_projet.ipynb
├── Rapport_Mini_Projet_IA.pdf
└── README.md
```

## Installation

```bash
pip install numpy matplotlib
```

## Utilisation

### Exécution complète
```bash
python run_all.py
```

### Exécution sélective
```bash
# Expériences E.1-E.4
python src/experiments.py

# Visualisation des trajectoires
python src/gen_traj_figures.py

# Analyses supplémentaires
python src/gen_extra_figures.py
```

## Expériences Réalisées

### E.1 : Comparaison des Algorithmes de Recherche
Évaluation comparative de UCS, Greedy Best-First et A* sur trois grilles de complexité croissante.

**Métriques** : coût du chemin, nœuds développés, taille maximale de OPEN

### E.2 : Impact de l'Incertitude
Analyse de l'effet du paramètre ε ∈ {0.0, 0.1, 0.2, 0.3} sur la performance réelle.

**Métriques** : probabilité d'atteindre le but, temps moyen d'atteinte, distribution des temps

### E.3 : Influence de l'Heuristique
Comparaison entre h=0 (équivalent à UCS) et heuristique Manhattan.

**Métriques** : réduction de l'exploration, maintien de l'optimalité

### E.4 : Weighted A*
Étude du compromis vitesse/optimalité avec différents poids w ∈ {1.0, 1.5, 2.0, 3.0, 5.0}.

**Métriques** : nœuds développés, ratio de sous-optimalité

## Résultats Principaux

### Efficacité de A*
Sur la grille difficile (12×12), A* développe 42 nœuds contre 87 pour UCS, tout en garantissant l'optimalité (coût = 24).

### Impact de l'Incertitude
Pour ε = 0.3 sur la grille moyenne, la probabilité d'atteindre le but chute à 30% malgré un plan optimal de coût 18.

### Validation du Modèle
L'écart entre le calcul matriciel (P^200) et la simulation Monte-Carlo (2000 trajectoires) reste inférieur à 2.5% pour tous les niveaux d'incertitude testés.

### Apériodicité
Identification de 27 états avec boucle propre (p_ii > 0) garantissant la convergence vers une distribution stationnaire.

## Figures Générées

Les résultats graphiques sont disponibles dans le dossier `figures/` :

- Chemins optimaux A* sur les trois grilles
- Comparaisons algorithmiques (barres, courbes)
- Évolution de π^(n)[GOAL] en fonction du temps
- Distributions des temps d'atteinte (histogrammes)
- Trajectoires stochastiques individuelles
- Heatmaps de densité de passage
- Visualisation de l'apériodicité

## Concepts Théoriques

### Admissibilité et Cohérence
L'heuristique Manhattan h(n) = |x - x_g| + |y - y_g| est :
- **Admissible** : h(n) ≤ h*(n) pour tout état n
- **Cohérente** : h(n) ≤ c(n,n') + h(n') pour toute transition

Ces propriétés garantissent l'optimalité de A* et évitent les réouvertures de nœuds.

### Modèle d'Incertitude
Pour une action voulue vers n' :
- Probabilité 1-ε : transition réussie vers n'
- Probabilité ε/2 : déviation latérale gauche
- Probabilité ε/2 : déviation latérale droite
- Si collision avec obstacle : l'agent reste sur place

### Analyse Markov
- **Matrice de transition** : P stochastique (somme des lignes = 1)
- **Évolution** : π^(n) = π^(0) × P^n
- **Absorption** : calcul des probabilités et temps moyens via la matrice fondamentale N = (I - Q)^(-1)

## Validation

Le code a été validé par :
1. Vérification de l'optimalité de A* (comparaison avec UCS)
2. Contrôle de la stochasticité de P (sommes des lignes)
3. Comparaison calcul analytique vs simulation Monte-Carlo
4. Tests sur trois grilles de complexité croissante

## Références

Ce projet s'appuie sur les concepts présentés dans :
- Synthèse "Chaînes de Markov à temps discret"
- Synthèse "Recherche heuristique : du Best-First à A*"

## Auteur

Projet réalisé dans le cadre du cours "Bases de l'Intelligence Artificielle"

---

**Note** : Le rapport détaillé (`Rapport_Mini_Projet_IA.pdf`) contient l'analyse complète des résultats, les tableaux comparatifs et la discussion des limites de l'approche.
