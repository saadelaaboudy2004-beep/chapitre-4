import numpy as np
import matplotlib.pyplot as plt


class MoleculeUNIFAC:
    """
    Cette classe permet de représenter une molécule
    selon la méthode UNIFAC, c’est-à-dire comme un
    assemblage de groupes fonctionnels.
    """

    def __init__(self, name, groups):
        """
        Initialisation de la molécule.

        name : nom de la molécule
        groups : dictionnaire {nom_du_groupe : nombre}
        Exemple : {'CH3': 1, 'CH2': 1, 'OH': 1}
        """
        self.name = name
        self.groups = groups

        # Paramètres structuraux UNIFAC des groupes
        # R : volume du groupe
        # Q : surface du groupe
        # Ces valeurs proviennent des tables UNIFAC
        self.R = {
            'CH3': 0.9011,
            'CH2': 0.6744,
            'CH': 0.4469,
            'OH': 1.0000,
            'H2O': 0.9200,
            'CH3CO': 1.6724
        }

        self.Q = {
            'CH3': 0.848,
            'CH2': 0.540,
            'CH': 0.228,
            'OH': 1.200,
            'H2O': 1.400,
            'CH3CO': 1.488
        }

        # Calcul des paramètres moléculaires r et q
        # r = somme(n_k * R_k)
        # q = somme(n_k * Q_k)
        # Ces grandeurs représentent la taille et la surface
        # globale de la molécule
        self.r = sum(n * self.R[g] for g, n in groups.items())
        self.q = sum(n * self.Q[g] for g, n in groups.items())

    def __repr__(self):
        return (
            f"Molecule ({self.name}) : r = {self.r:.4f}, "
            f"q = {self.q:.4f}, groupes = {self.groups}"
        )


def unifac_combinatorial(x1, mol1, mol2):
    """
    Calcul de la contribution combinatorielle
    des coefficients d’activité selon UNIFAC.

    D’après le cours :
    ln(γi) = ln(γi^comb) + ln(γi^res)

    Ici, on calcule uniquement ln(γi^comb),
    qui traduit les effets entropiques
    (taille et forme des molécules).
    """

    x2 = 1 - x1
    z = 10  # nombre de coordination (constante UNIFAC)

    # Paramètres moléculaires
    r1, r2 = mol1.r, mol2.r
    q1, q2 = mol1.q, mol2.q

    # Fractions de volume φ_i
    # φ_i = x_i r_i / Σ(x_j r_j)
    phi1 = x1 * r1 / (x1 * r1 + x2 * r2)
    phi2 = x2 * r2 / (x1 * r1 + x2 * r2)

    # Fractions de surface θ_i
    # θ_i = x_i q_i / Σ(x_j q_j)
    theta1 = x1 * q1 / (x1 * q1 + x2 * q2)
    theta2 = x2 * q2 / (x1 * q1 + x2 * q2)

    # Paramètres l_i (définis par la théorie UNIFAC)
    l1 = (z / 2) * (r1 - q1) - (r1 - 1)
    l2 = (z / 2) * (r2 - q2) - (r2 - 1)

    # Expression de ln(γi^comb)
    # Issue directement du développement thermodynamique
    # du modèle UNIQUAC/UNIFAC
    ln_gamma1_comb = (
        np.log(phi1 / x1)
        + (z / 2) * q1 * np.log(theta1 / phi1)
        + l1
        - (phi1 / x1) * (x1 * l1 + x2 * l2)
    )

    ln_gamma2_comb = (
        np.log(phi2 / x2)
        + (z / 2) * q2 * np.log(theta2 / phi2)
        + l2
        - (phi2 / x2) * (x1 * l1 + x2 * l2)
    )

    return ln_gamma1_comb, ln_gamma2_comb


# Définition des molécules à partir des groupes UNIFAC

# Éthanol : CH3–CH2–OH
ethanol = MoleculeUNIFAC(
    "Ethanol",
    {'CH3': 1, 'CH2': 1, 'OH': 1}
)

# Eau : H2O
eau = MoleculeUNIFAC(
    "Eau",
    {'H2O': 1}
)

# Acétone : CH3–CO–CH3
acetone = MoleculeUNIFAC(
    "Acetone",
    {'CH3': 2, 'CH3CO': 1}
)

# Affichage des paramètres moléculaires
print("Paramètres moléculaires UNIFAC")
print("=" * 50)
print(ethanol)
print(eau)
print(acetone)
print()


# Étude du système Éthanol (1) – Eau (2)
x1_range = np.linspace(0.01, 0.99, 99)

ln_gamma1_comb, ln_gamma2_comb = unifac_combinatorial(
    x1_range, ethanol, eau
)

gamma1_comb = np.exp(ln_gamma1_comb)
gamma2_comb = np.exp(ln_gamma2_comb)

# Affichage numérique
print("Contribution combinatorielle UNIFAC – Ethanol / Eau")
print("=" * 60)
print(f"{'x_ethanol':<12} {'gamma1_comb':<15} {'gamma2_comb':<15}")
print("-" * 60)

for i in [0, 24, 49, 74, 98]:
    print(
        f"{x1_range[i]:<12.3f} "
        f"{gamma1_comb[i]:<15.4f} "
        f"{gamma2_comb[i]:<15.4f}"
    )

# Représentation graphique
plt.figure(figsize=(10, 6))

plt.plot(
    x1_range, gamma1_comb,
    'b-', linewidth=2,
    label='Ethanol – partie combinatorielle'
)

plt.plot(
    x1_range, gamma2_comb,
    'r-', linewidth=2,
    label='Eau – partie combinatorielle'
)

plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)

plt.xlabel("Fraction molaire d’éthanol, x₁")
plt.ylabel("Coefficient d’activité (partie combinatorielle)")
plt.title(
    "UNIFAC – Contribution combinatorielle\nSystème Éthanol – Eau",
    fontweight='bold'
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("unifac_combinatorial.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nNote :")
print("Le calcul complet UNIFAC nécessite la contribution résiduelle,")
print("basée sur les paramètres d’interaction entre groupes")
print("(tables UNIFAC – DECHEMA).")