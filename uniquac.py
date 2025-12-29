import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson


def uniquac_binary(x1, T, r1, r2, q1, q2, a12, a21):
    """
    Cette fonction calcule les coefficients d’activité gamma1 et gamma2
    d’un mélange binaire en utilisant le modèle UNIQUAC.

    Selon le cours, le modèle UNIQUAC décompose :
    ln(gammai) = contribution combinatoire + contribution résiduelle
    """

    # Le nombre de coordination z est une constante du modèle UNIQUAC
    # Il représente le nombre moyen de molécules voisines autour d’une molécule
    z = 10

    # La fraction molaire du constituant 2 est obtenue par la relation x1 + x2 = 1
    x2 = 1 - x1


    # PARTIE COMBINATOIRE
    """
    Cette partie décrit les effets entropiques dus aux différences 
    de taille et de forme des molécules dans le mélange
    """
    # Les fractions de volume phi_i tiennent compte du volume moléculaire r_i
    # Elles représentent la fraction de volume occupée par chaque constituant
    phi1 = x1 * r1 / (x1 * r1 + x2 * r2)
    phi2 = x2 * r2 / (x1 * r1 + x2 * r2)

    # Les fractions de surface theta_i tiennent compte de la surface moléculaire q_i
    # Elles représentent la fraction de surface de contact de chaque constituant
    theta1 = x1 * q1 / (x1 * q1 + x2 * q2)
    theta2 = x2 * q2 / (x1 * q1 + x2 * q2)

    # Les paramètres l_i sont des termes correctifs du modèle UNIQUAC
    # Ils dépendent de la taille et de la surface des molécules
    l1 = (z / 2) * (r1 - q1) - (r1 - 1)
    l2 = (z / 2) * (r2 - q2) - (r2 - 1)

    # Calcul de la contribution combinatoire au coefficient d’activité
    # Cette expression provient directement de la formulation théorique UNIQUAC
    ln_gamma1_comb = (np.log(phi1 / x1)+ (z / 2) * q1 * np.log(theta1 / phi1)
        + l1- (phi1 / x1) * (x1 * l1 + x2 * l2))

    ln_gamma2_comb = (np.log(phi2 / x2)+ (z / 2) * q2 * np.log(theta2 / phi2)
        + l2- (phi2 / x2) * (x1 * l1 + x2 * l2))


   
    # PARTIE RÉSIDUELLE
  
    # Cette partie décrit les interactions énergétiques entre molécules
    # Elle dépend des paramètres énergétiques a_ij

    # Les paramètres tau_ij représentent l’énergie d’interaction entre i et j
    # Ils sont définis par tau_ij = exp( -a_ij / T )
    tau12 = np.exp(-a12 / T)
    tau21 = np.exp(-a21 / T)

    # Contribution résiduelle au coefficient d’activité
    # Elle traduit l’effet des interactions spécifiques entre constituants
    ln_gamma1_res = q1 * (1- np.log(theta1 + theta2 * tau21)
        - theta1 / (theta1 + theta2 * tau21)
        - theta2 * tau12 / (theta2 + theta1 * tau12))

    ln_gamma2_res = q2 * (1- np.log(theta2 + theta1 * tau12)
        - theta2 / (theta2 + theta1 * tau12)
        - theta1 * tau21 / (theta1 + theta2 * tau21))


   
    # COEFFICIENTS D’ACTIVITÉ TOTAUX
    
    # Le coefficient d’activité total est obtenu en additionnant
    # la contribution combinatoire et la contribution résiduelle
    ln_gamma1 = ln_gamma1_comb + ln_gamma1_res
    ln_gamma2 = ln_gamma2_comb + ln_gamma2_res

    # On repasse de ln(gamma) à gamma par exponentiation
    gamma1 = np.exp(ln_gamma1)
    gamma2 = np.exp(ln_gamma2)

    return gamma1, gamma2


# Le système étudié est : Acétone (1) – Chloroforme (2)
# La température du mélange est fixée à 50 °C
T = 323.15  # Température en Kelvin

# Les paramètres r et q décrivent respectivement le volume
# et la surface des molécules 
r1, q1 = 2.574, 2.336   # Acétone
r2, q2 = 2.870, 2.410   # Chloroforme

# Les paramètres a12 et a21 représentent les interactions énergétiques
# entre les deux constituants
a12 = -52.39
a21 = 340.35


# On étudie le comportement du mélange sur toute la gamme de composition
x1_range = np.linspace(0.001, 0.999, 100)

# Calcul des coefficients d’activité pour chaque composition
gamma1_vals, gamma2_vals = uniquac_binary(
    x1_range, T, r1, r2, q1, q2, a12, a21)


# Affichage de quelques valeurs numériques représentatives
print("Résultats UNIQUAC pour Acétone (1) – Chloroforme (2) à 50 °C")
print("=" * 60)
print(f"{'x_acetone':<12} {'gamma_acetone':<15} {'gamma_chloroforme':<15}")
print("-" * 60)

for i in [0, 24, 49, 74, 99]:
    print(f"{x1_range[i]:<12.3f} {gamma1_vals[i]:<15.4f} {gamma2_vals[i]:<15.4f}")


# Vérification de la relation de Gibbs–Duhem
# Pour un modèle thermodynamique cohérent, la différence des aires doit être nulle
area1 = simpson(np.log(gamma1_vals) * x1_range, x1_range)
area2 = simpson(np.log(gamma2_vals) * (1 - x1_range), x1_range)

print("\nVérification de Gibbs–Duhem :")
print(f"Aire ln(γ1) = {area1:.6f}")
print(f"Aire ln(γ2) = {area2:.6f}")
print(f"Différence = {abs(area1 - area2):.6f} (doit être proche de 0)")


# Tracé des coefficients d’activité et de l’énergie de Gibbs excédentaire
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Les coefficients d’activité montrent l’écart à l’idéalité (γ = 1)
ax1.plot(x1_range, gamma1_vals, 'b-', linewidth=2, label='Acétone')
ax1.plot(x1_range, gamma2_vals, 'r-', linewidth=2, label='Chloroforme')
ax1.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel("Fraction molaire d’acétone, x₁")
ax1.set_ylabel("Coefficient d’activité, γ")
ax1.set_title("UNIQUAC – Coefficients d’activité")
ax1.legend()
ax1.grid(True, alpha=0.3)

# L’énergie de Gibbs excédentaire permet de quantifier la non-idéalité du mélange
# Gᴱ / RT = x1 x2 ( lnγ1 + lnγ2 )
GE_RT = x1_range * (1 - x1_range) * (
    np.log(gamma1_vals) + np.log(gamma2_vals))

ax2.plot(x1_range, GE_RT, 'g-', linewidth=2)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.set_xlabel("Fraction molaire d’acétone, x₁")
ax2.set_ylabel(r"$G^E / RT$")
ax2.set_title("Énergie de Gibbs excédentaire")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()