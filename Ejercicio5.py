import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, ks_2samp, chi2_contingency

def sample_normal_inverse(mu, sigma, size, seed=None):
    
    rng = np.random.default_rng(seed)
    u = rng.random(size)
    return norm.ppf(u, loc=mu, scale=sigma)

def bins_by_theoretical_quantiles(mu, sigma, N, min_esperado=5, max_bins=50):
    
    K = max(2, min(max_bins, N // min_esperado))
    qs = np.linspace(0, 1, K + 1)[1:-1]
    internal_edges = norm.ppf(qs, loc=mu, scale=sigma)
    edges = np.concatenate(([-np.inf], internal_edges, [np.inf]))
    return edges

def comparar_normal(mu=0.0, sigma=1.0, N=10000, seed=42, alpha=0.05, show_plot=True):
    # generar muestras
    theo = norm.rvs(loc=mu, scale=sigma, size=N, random_state=seed)
    emp = sample_normal_inverse(mu, sigma, N, seed=seed+1)

    # estadísticas descriptivas
    mean_theo = np.mean(theo); mean_emp = np.mean(emp)
    var_theo = np.var(theo, ddof=1); var_emp = np.var(emp, ddof=1)

    print("="*70)
    print("GENERACIÓN DE MUESTRAS (Normal)")
    print("="*70)
    print(f"mu = {mu}, sigma = {sigma}, N = {N}, seed = {seed}")
    print(f"Media (scipy): {mean_theo:.6f} | Media (inv): {mean_emp:.6f} | Esperada: {mu:.6f}")
    print(f"Var   (scipy): {var_theo:.6f} | Var   (inv): {var_emp:.6f} | Esperada: {sigma**2:.6f}")

    # gráfica comparativa
    if show_plot:
        plt.figure(figsize=(10,5))
        bins = 100
        plt.hist(theo, bins=bins, density=True, alpha=0.5, label='Muestra Teórica (scipy.stats)')
        plt.hist(emp, bins=bins, density=True, alpha=0.5, label='Muestra Empírica (Transformada inv.)', histtype='step', linewidth=2)
        xs = np.linspace(np.percentile(np.concatenate([theo, emp]), 0.5),
                         np.percentile(np.concatenate([theo, emp]), 99.5), 400)
        plt.plot(xs, norm.pdf(xs, loc=mu, scale=sigma), 'k--', label='PDF teórica', lw=1)
        plt.xlabel('x'); plt.ylabel('Densidad'); plt.title(f'Comparación de muestras N({mu},{sigma**2})')
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    # Prueba KS (dos muestras)
    ks_stat, ks_pvalue = ks_2samp(theo, emp)

    print("\n" + "="*70)
    print("PRUEBA DE KOLMOGOROV-SMIRNOV (dos muestras)")
    print("="*70)
    print(f"Estadístico KS: {ks_stat:.6f}")
    print(f"p-value KS:     {ks_pvalue:.6f}")
    print(f"Nivel α:        {alpha}")
    if ks_pvalue > alpha:
        print("Conclusión KS: NO rechazamos H0 → Las muestras pueden provenir de la misma distribución.")
    else:
        print("Conclusión KS: Rechazamos H0 → Las muestras parecen provenir de distribuciones diferentes.")

    # Prueba Chi-Cuadrado: construir bins con esperados suficientes
    edges = bins_by_theoretical_quantiles(mu, sigma, N, min_esperado=5, max_bins=50)
    counts_theo, _ = np.histogram(theo, bins=edges)
    counts_emp, _ = np.histogram(emp, bins=edges)

    tabla = np.vstack([counts_theo, counts_emp])
    chi2_stat, chi2_pvalue, chi2_dof, chi2_exp = chi2_contingency(tabla)

    print("\n" + "="*70)
    print("PRUEBA DE CHI-CUADRADO (contingencia entre bins definidos por cuantiles teóricos)")
    print("="*70)
    print(f"Estadístico χ²: {chi2_stat:.6f}")
    print(f"p-value χ²:     {chi2_pvalue:.6f}")
    print(f"Grados libertad: {chi2_dof}")
    print(f"Nivel α:         {alpha}")
    if chi2_pvalue > alpha:
        print("Conclusión χ²: NO rechazamos H0 → Las frecuencias por bins son consistentes entre muestras.")
    else:
        print("Conclusión χ²: Rechazamos H0 → Las frecuencias difieren significativamente entre muestras.")

    return {
        'mu': mu, 'sigma': sigma, 'N': N, 'seed': seed, 'alpha': alpha,
        'theo': theo, 'emp': emp,
        'ks_stat': ks_stat, 'ks_pvalue': ks_pvalue,
        'chi2_stat': chi2_stat, 'chi2_pvalue': chi2_pvalue,
        'chi2_dof': chi2_dof, 'tabla_chi2': tabla, 'edges': edges
    }

# Parámetros del ejercicio
mu = 0.0
sigma = 1.0
N = 10000
alpha = 0.05

# Ejecutar comparación
resultados = comparar_normal(mu=mu, sigma=sigma, N=N, seed=42, alpha=alpha, show_plot=True)