import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

# ======== GENERADOR MERSENNE TWISTER ========
def generar_mersenne(N, seed=None):
    """Genera N números uniformes U(0,1) usando Mersenne Twister."""
    rng = np.random.default_rng(seed)  # usa Mersenne Twister por defecto
    return rng.random(N)

# ======== PARÁMETROS ========
N = 10000  # tamaño de muestra
seed = 42  # semilla opcional (para reproducibilidad)

# ======== GENERAR MUESTRA ========
u = generar_mersenne(N, seed)

# ======== ESTADÍSTICOS ========
print("--- Estadísticos ---")
print(f"Media: {np.mean(u):.4f}")
print(f"Varianza: {np.var(u):.4f}")
print(f"Mínimo: {np.min(u):.4f}")
print(f"Máximo: {np.max(u):.4f}")

# ======== HISTOGRAMA ========
plt.figure(figsize=(6,4))
plt.hist(u, bins=20, color='lightgreen', edgecolor='black', alpha=0.8)
plt.title("Histograma - Generador Mersenne Twister (Unif(0,1))")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.show()

# ======== PRUEBA DE HIPÓTESIS (Kolmogorov–Smirnov) ========
print("\n--- Prueba K-S ---")
ks = kstest(u, 'uniform')
print("Estadístico D:", ks.statistic)
print("Valor p:", ks.pvalue)

# ======== INTERPRETACIÓN ========
alpha = 0.05
if ks.pvalue > alpha:
    print("✅ No se rechaza H₀: la muestra proviene de una distribución uniforme.")
else:
    print("❌ Se rechaza H₀: la muestra NO parece uniforme.")
