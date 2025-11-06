import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom, ks_2samp, chi2_contingency

def sample_geom_inverse(p, size, seed=None):
    
    rng = np.random.default_rng(seed)
    u = rng.random(size)
    ratios = np.log1p(-u) / np.log1p(-p)  # log(1-u)/log(1-p)
    ks = np.ceil(ratios).astype(int)
    ks[ks == 0] = 1
    return ks

def agrupar_por_esperado(obs1, obs2, p, N, min_esperado=5):
    
    # rango considerado (incluye cola razonable)
    max_k = int(max(np.max(obs1), np.max(obs2), geom.ppf(0.999, p))) + 1
    ks = np.arange(1, max_k + 1)
    pmf = geom.pmf(ks, p)
    esperados = N * pmf

    conteos1 = np.array([np.sum(obs1 == k) for k in ks])
    conteos2 = np.array([np.sum(obs2 == k) for k in ks])

    grupos1 = []
    grupos2 = []

    acum_exp = 0.0
    acum_c1 = 0
    acum_c2 = 0

    for e, c1, c2 in zip(esperados, conteos1, conteos2):
        acum_exp += e
        acum_c1 += c1
        acum_c2 += c2
        if acum_exp >= min_esperado:
            grupos1.append(acum_c1)
            grupos2.append(acum_c2)
            acum_exp = 0.0
            acum_c1 = 0
            acum_c2 = 0

    # si queda residuo, agregarlo al último grupo existente (o crear uno si no existe)
    if acum_c1 + acum_c2 > 0:
        if len(grupos1) > 0:
            grupos1[-1] += acum_c1
            grupos2[-1] += acum_c2
        else:
            grupos1.append(acum_c1)
            grupos2.append(acum_c2)

    return np.array(grupos1), np.array(grupos2)

def comparar_geom(p=0.3, N=10000, seed=42, alpha=0.05, show_plot=True):
    # semillas para reproducibilidad separadas
    theo = geom.rvs(p, size=N, random_state=seed)
    emp = sample_geom_inverse(p, N, seed=seed + 1)

    # Estadísticas descriptivas
    media_teo = np.mean(theo)
    media_emp = np.mean(emp)
    var_teo = np.var(theo, ddof=1)
    var_emp = np.var(emp, ddof=1)

    print("="*70)
    print("GENERACIÓN DE MUESTRAS")
    print("="*70)
    print(f"p = {p}, N = {N}, seed = {seed}")
    print(f"Media teórica esperada: {1/p:.4f}")
    print(f"Media muestra (scipy):  {media_teo:.4f}")
    print(f"Media muestra (inv):    {media_emp:.4f}")
    print(f"Varianza teórica esperada: {(1-p)/p**2:.4f}")
    print(f"Varianza muestra (scipy):  {var_teo:.4f}")
    print(f"Varianza muestra (inv):    {var_emp:.4f}")

    # Gráfica comparativa (histograma + PMF teórica)
    if show_plot:
        max_k = int(max(np.max(theo), np.max(emp), geom.ppf(0.999, p))) + 1
        bins = np.arange(0.5, max_k + 1.5, 1)
        plt.figure(figsize=(10,5))
        plt.hist(theo, bins=bins, density=True, alpha=0.6, label='Muestra Teórica (scipy.stats)')
        plt.hist(emp, bins=bins, density=True, alpha=0.6, label='Muestra Empírica (Transformada Inv.)', histtype='step', linewidth=2)
        ks_plot = np.arange(1, max_k + 1)
        plt.plot(ks_plot, geom.pmf(ks_plot, p), 'ko-', label='PMF teórica', ms=4)
        plt.xlabel('k')
        plt.ylabel('Frecuencia relativa / Probabilidad')
        plt.title(f'Comparación de muestras Geom(p={p})')
        plt.legend()
        plt.grid(alpha=0.4, linestyle='--')
        plt.tight_layout()
        plt.show()

    # Prueba Kolmogorov-Smirnov (dos muestras)
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

    # Prueba Chi-Cuadrado (frecuencias agrupadas)
    grupos_teo, grupos_emp = agrupar_por_esperado(theo, emp, p, N, min_esperado=5)

    # Construir tabla 2 x m
    tabla = np.vstack([grupos_teo, grupos_emp])
    chi2_stat, chi2_pvalue, chi2_dof, chi2_exp = chi2_contingency(tabla)

    print("\n" + "="*70)
    print("PRUEBA DE CHI-CUADRADO (contingencia entre frecuencias agrupadas)")
    print("="*70)
    print(f"Estadístico χ²: {chi2_stat:.6f}")
    print(f"p-value χ²:     {chi2_pvalue:.6f}")
    print(f"Grados libertad: {chi2_dof}")
    print(f"Nivel α:         {alpha}")
    if chi2_pvalue > alpha:
        print("Conclusión χ²: NO rechazamos H0 → Las frecuencias son consistentes entre ambas muestras.")
    else:
        print("Conclusión χ²: Rechazamos H0 → Las frecuencias difieren significativamente entre muestras.")

    return {
        'p': p, 'N': N, 'seed': seed, 'alpha': alpha,
        'theo': theo, 'emp': emp,
        'ks_stat': ks_stat, 'ks_pvalue': ks_pvalue,
        'chi2_stat': chi2_stat, 'chi2_pvalue': chi2_pvalue,
        'chi2_dof': chi2_dof, 'tabla_chi2': tabla
    }

# Parámetros del ejercicio
p = 0.3
N = 10000
alpha = 0.05

# Ejecutar la comparación
resultados = comparar_geom(p=p, N=N, seed=42, alpha=alpha, show_plot=True)