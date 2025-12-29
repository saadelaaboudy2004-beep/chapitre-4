# numpy : calcul numérique (vecteurs, exponentielle, etc.)
# matplotlib : tracé des courbes
import numpy as np
import matplotlib.pyplot as plt



# FONCTION NRTL POUR UN SYSTÈME BINAIRE

def nrtl_binary(x1, T, a12, a21, alpha):
    """
    Cette fonction calcule les coefficients d’activité gamma1 et gamma2
    d’un mélange binaire à l’aide du modèle NRTL.

    """

    # Fraction molaire du constituant 2 (relation de fermeture)
    x2 = 1 - x1

    
    # CALCUL DES PARAMÈTRES τ (tau)
 
    # D’après le cours :
    # τ_ij = a_ij / T
    # Ces termes représentent l’interaction énergétique
    # entre les constituants i et j
    tau12 = a12 / T
    tau21 = a21 / T

  
    # CALCUL DES PARAMÈTRES G_ij
    
    # G_ij = exp( -α * τ_ij )
    # Ils traduisent le degré de non-randomness du mélange
    
    G12 = np.exp(-alpha * tau12)
    G21 = np.exp(-alpha * tau21)


    # CALCUL DE ln(gamma1)
   
    # Expression théorique NRTL (vue dans le cours)
    # ln(gamma1) dépend des interactions avec le constituant 2
    # et est pondérée par x2²
    term1_gamma1 = tau21 * (G21 / (x1 + x2 * G21))**2
    term2_gamma1 = tau12 * G12 / (x2 + x1 * G12)**2
    ln_gamma1 = x2**2 * (term1_gamma1 + term2_gamma1)

  
    # CALCUL DE ln(gamma2)
 
    # Formule symétrique à celle de gamma1
    # ln(gamma2) est pondéré par x1²
    term1_gamma2 = tau12 * (G12 / (x2 + x1 * G12))**2
    term2_gamma2 = tau21 * G21 / (x1 + x2 * G21)**2
    ln_gamma2 = x1**2 * (term1_gamma2 + term2_gamma2)

    
    # PASSAGE DE ln(γ) À γ
    
    # Le coefficient d’activité est obtenu par exponentiation
    gamma1 = np.exp(ln_gamma1)
    gamma2 = np.exp(ln_gamma2)

    return gamma1, gamma2



# PARAMÈTRES DU SYSTÈME ÉTUDIÉ

# Système binaire : Éthanol (1) – Eau (2)
# Valeurs issues de données expérimentales 
T = 298.15       # Température en Kelvin (25 °C)
a12 = -0.8009    # Paramètre énergétique éthanol → eau
a21 = 1239.5     # Paramètre énergétique eau → éthanol
alpha = 0.3      # Paramètre de non-randomness 




# On étudie le comportement du mélange de x1 ≈ 0 à x1 ≈ 1
x1_range = np.linspace(0.001, 0.999, 100)

# Calcul des coefficients d’activité pour chaque composition
gamma1_vals, gamma2_vals = nrtl_binary(x1_range, T, a12, a21, alpha)


# AFFICHAGE NUMÉRIQUE DES RÉSULTATS

# Affichage de quelques points représentatifs
print("Résultats NRTL pour Éthanol (1) – Eau (2) à T = 25 °C")
print("=" * 60)
print(f"{'x_ethanol':<12} {'gamma_ethanol':<15} {'gamma_eau':<15}")
print("-" * 60)

for i in [0, 24, 49, 74, 99]:
    print(f"{x1_range[i]:<12.3f} {gamma1_vals[i]:<15.4f} {gamma2_vals[i]:<15.4f}")



# COEFFICIENTS À DILUTION INFINIE

# D’après le cours :
# gamma1inf : comportement de l’éthanol très dilué dans l’eau
# gamma2inf : comportement de l’eau très diluée dans l’éthanol
gamma1_inf = nrtl_binary(0.001, T, a12, a21, alpha)[0]
gamma2_inf = nrtl_binary(0.999, T, a12, a21, alpha)[1]

print("\nCoefficients à dilution infinie :")
print(f"gamma_1^inf (éthanol dans eau) = {gamma1_inf:.4f}")
print(f"gamma_2^inf (eau dans éthanol) = {gamma2_inf:.4f}")



# REPRÉSENTATION GRAPHIQUE

# Le graphe montre l’écart à l’idéalité (γ = 1 → solution idéale)
plt.figure(figsize=(10, 6))
plt.plot(x1_range, gamma1_vals, 'b-', linewidth=2, label='Éthanol')
plt.plot(x1_range, gamma2_vals, 'r-', linewidth=2, label='Eau')

# Ligne de référence γ = 1 (solution idéale)
plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)

plt.xlabel("Fraction molaire d’éthanol, x", fontsize=12)
plt.ylabel("Coefficient d’activité, γ", fontsize=12)
plt.title("Coefficients d’activité – Système Éthanol–Eau (NRTL, 25 °C)",
          fontsize=13, fontweight='bold')

plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, max(gamma1_inf, gamma2_inf) * 1.1)

plt.tight_layout()
plt.savefig("nrtl_ethanol_eau.png", dpi=300, bbox_inches="tight")
plt.show()