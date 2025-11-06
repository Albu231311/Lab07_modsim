import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

# ======== GENERADOR LCG ========
def lcg(a, c, m, seed, N):
    """Generador lineal congruencial (LCG)."""
    x = np.zeros(N)
    x[0] = seed
    for i in range(1, N):
        x[i] = (a * x[i-1] + c) % m
    return x / m  # Normaliza para tener valores en [0, 1)

# ======== PARÁMETROS ========
# Conjunto 1
params1 = {'a': 1103515245, 'c': 12345, 'm': 2**31, 'seed': 42, 'N': 10000}
# Conjunto 2
params2 = {'a': 1664525, 'c': 1013904223, 'm': 2**32, 'seed': 123, 'N': 10000}

# ======== GENERAR MUESTRAS ========
u1 = lcg(**params1)
u2 = lcg(**params2)

# ======== ESTADÍSTICOS ========
def estadisticos(u):
    print(f"Media: {np.mean(u):.4f}")
    print(f"Varianza: {np.var(u):.4f}")
    print(f"Mínimo: {np.min(u):.4f}")
    print(f"Máximo: {np.max(u):.4f}")

print("\n--- PARÁMETROS 1 ---")
estadisticos(u1)
print("\n--- PARÁMETROS 2 ---")
estadisticos(u2)

# ======== HISTOGRAMAS ========
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(u1, bins=20, color='skyblue', edgecolor='black')
plt.title("Histograma - Parámetros 1")
plt.subplot(1,2,2)
plt.hist(u2, bins=20, color='salmon', edgecolor='black')
plt.title("Histograma - Parámetros 2")
plt.show()

# ======== PRUEBA DE HIPÓTESIS (Kolmogorov–Smirnov) ========
print("\n--- Prueba K-S ---")
ks1 = kstest(u1, 'uniform')
ks2 = kstest(u2, 'uniform')
print("Parámetros 1:", ks1)
print("Parámetros 2:", ks2)
