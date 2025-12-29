# ================================
# IMPORT DES BIBLIOTHÈQUES
# ================================

# numpy : calculs numériques (log, exp, tableaux, etc.)
import numpy as np

# matplotlib : tracé des courbes
import matplotlib.pyplot as plt


# ================================
# CLASSE MOLÉCULE (MÉTHODE UNIFAC)
# ================================

class MoleculeUNIFAC:
    """
    Cette classe permet de représenter une molécule
    selon la méthode UNIFAC, c’est-à-dire comme un
    assemblage de groupes fonctionnels.
    """

    def __init__(self, name, groups):
        """
        Initialisation de la molécule

        name   : nom de la molécule
        groups : dictionnaire {groupe : nombre}
                 Exemple : {'CH3':1, 'CH2':1, 'OH':1}
        """
        self.name = name
        self.groups = groups

        # Paramètres UNIFAC R (volume) pour chaque groupe
        # Ces valeurs proviennent des tables UNIFAC
        self.R = {
            'CH3': 0.9011,
            'CH2': 0.6744,
            'CH': 0.4469,
            'OH': 1.0000,
            'H2O': 0.9200,
            'CH3CO': 1.6724
        }

        # Paramètres UNIFAC Q (surface)
        self.Q = {
            'CH3': 0.848,
            'CH2': 0.540,
            'CH': 0.228,
            'OH': 1.200,
            'H2O': 1.400,
            'CH3CO': 1.488
        }

        # Calcul du paramètre r de la molécule
        # r = Σ (n_k × R_k)
        self.r = sum(n * self.R[g] for g, n in groups.items())

        # Calcul du paramètre q de la molécule
        # q = Σ (n_k × Q_k)
        self.q = sum(n * self.Q[g] for g, n in groups.items())

    def __repr__(self):
        """
        Affichage lisible de la molécule
        """
        return f"{self.name} : r = {self.r:.4f}, q = {self.q:.4f}"


# ==========================================
# FONCTION UNIFAC – PARTIE COMBINATORIELLE
# ==========================================

def unifac_combinatorial(x1, mol1, mol2):
    """
    Calcul de la contribution combinatorielle
    des coefficients d’activité selon UNIF
