# #!/usr/bin/env python3
# import argparse
# import numpy as np
# import pandas as pd

# #mocap data
# mocap_marker_indices = [2, 0, 1, 7, 8, 10, 11, 12, 13, 14, 15, 18, 19, 16, 17, 3, 4, 5, 6]
# #hpe data
# hpe_marker_indices = [18, 6, 5, 12, 11, 14, 13, 16, 15, 25, 24, 23, 22, 21, 20, 8, 7, 10, 9]


# def main():
#     ap = argparse.ArgumentParser(description="Estimation sigma bruit blanc via corrélation x (fiable) vs y (bruité).")
#     ap.add_argument("x_csv", help="CSV du signal fiable (colonnes numériques alignées)")
#     ap.add_argument("y_csv", help="CSV du signal bruité (mêmes colonnes et longueur)")
#     ap.add_argument("--sep", default=",", help="Séparateur CSV (défaut: ,)")
#     ap.add_argument("--skiprows", type=int, default=0, help="Lignes à ignorer en tête")
#     ap.add_argument("--cols", nargs="*", help="Liste explicite de colonnes à utiliser (sinon: toutes numériques communes)")
#     ap.add_argument("--dropna", action="store_true", help="Supprimer lignes avec NaN (sinon remplissage par interpolation simple)")
#     args = ap.parse_args()

#     # Chargement
#     xdf = pd.read_csv(args.x_csv, sep=args.sep, skiprows=args.skiprows)
#     ydf = pd.read_csv(args.y_csv, sep=args.sep, skiprows=args.skiprows)

#     xdf = xdf[xdf.columns[mocap_marker_indices]]
#     ydf = ydf[ydf.columns[mocap_marker_indices]]

#     # Colonnes cibles
#     if args.cols:
#         cols = [c for c in args.cols if c in xdf.columns and c in ydf.columns]
#         if not cols:
#             raise SystemExit("Aucune colonne demandée n'existe dans les deux fichiers.")
#         xdf = xdf[cols]
#         ydf = ydf[cols]
#     else:
#         xnum = xdf.select_dtypes(include=[np.number])
#         ynum = ydf.select_dtypes(include=[np.number])
#         cols = list(xnum.columns)
#         # cols = list(set(xnum.columns).intersection(ynum.columns))
#         # if not cols:
#         #     raise SystemExit("Pas de colonnes numériques communes entre les deux CSV.")
#         # xdf = xdf[cols]
#         # ydf = ydf[cols]

#     # Alignement longueur
#     n = min(len(xdf), len(ydf))
#     x = xdf.iloc[:n].to_numpy(dtype=float)
#     y = ydf.iloc[:n].to_numpy(dtype=float)

#     # Gestion des NaN
#     if args.dropna:
#         mask = ~np.isnan(x).any(axis=1) & ~np.isnan(y).any(axis=1)
#         x, y = x[mask], y[mask]
#     else:
#         # interpolation simple colonne par colonne
#         def interp_nan(a):
#             s = pd.Series(a)
#             s = s.interpolate(limit_direction="both")
#             return s.to_numpy()
#         x = np.column_stack([interp_nan(x[:, i]) for i in range(x.shape[1])])
#         y = np.column_stack([interp_nan(y[:, i]) for i in range(y.shape[1])])

#     # Centrer (corrélation = covariance des versions centrées / std)
#     x_c = x - x.mean(axis=0, keepdims=True)
#     y_c = y - y.mean(axis=0, keepdims=True)

#     # Écarts-types
#     sx = x_c.std(axis=0, ddof=1)
#     sy = y_c.std(axis=0, ddof=1)

#     # Corrélation Pearson à lag 0 par colonne
#     # rho = cov(x,y)/(sx*sy) = (mean(x_c*y_c))/(sx*sy) avec ddof cohérent
#     cov_xy = (x_c * y_c).sum(axis=0) / (len(x_c) - 1)  # covariance (ddof=1)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         rho = cov_xy / (sx * sy)
#     rho = np.clip(rho, -1.0, 1.0)

#     # Estimation sigma_n par colonne : sy * sqrt(1 - rho^2)
#     sigma_n = sy * np.sqrt(np.maximum(0.0, 1.0 - rho**2))

#     # Résumés
#     def fmt(v): return ", ".join(f"{x:.6g}" for x in v)
#     print(f"Colonnes utilisées ({len(cols)}): {', '.join(cols)}\n")

#     print("rho (corrélation x vs y) par colonne:")
#     print(fmt(rho))
#     print("\nσ_y (std signal bruité) par colonne:")
#     print(fmt(sy))
#     print("\nσ_n estimé (std bruit) par colonne:")
#     print(fmt(sigma_n))

#     print("\n--- Agrégats ---")
#     print(f"rho moyen        : {np.nanmean(rho):.6g} (médiane {np.nanmedian(rho):.6g})")
#     print(f"sigma_n moyen    : {np.nanmean(sigma_n):.6g} (médiane {np.nanmedian(sigma_n):.6g})")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd

#mocap data
mocap_marker_indices = [2, 0, 1, 7, 8, 10, 11, 12, 13, 14, 15, 18, 19, 16, 17, 3, 4, 5, 6]
#hpe data
hpe_marker_indices = [18, 6, 5, 12, 11, 14, 13, 16, 15, 25, 24, 23, 22, 21, 20, 8, 7, 10, 9]


if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <csv_fiable> <csv_bruite>")
    sys.exit(1)

x = pd.read_csv(sys.argv[1]).select_dtypes(np.number)
y = pd.read_csv(sys.argv[2]).select_dtypes(np.number)

x = x[x.columns[mocap_marker_indices]]
y = y[y.columns[mocap_marker_indices]]

# # Colonnes communes numériques
# cols = list(set(x.columns).intersection(y.columns))
# if not cols:
#     raise SystemExit("Aucune colonne numérique commune.")
cols = list(set(x.columns))
x = x.to_numpy(dtype=float)
y = y.to_numpy(dtype=float)

x = x / 1000.0


# Alignement longueur
n = min(len(x), len(y))
x = x[:n]; y = y[:n]

# Centrage
x_c = x - x.mean(axis=0, keepdims=True)
y_c = y - y.mean(axis=0, keepdims=True)

# Écarts-types (ddof=1)
sx = x_c.std(axis=0, ddof=1)
sy = y_c.std(axis=0, ddof=1)

# Corrélation à lag 0 par colonne
cov_xy = (x_c * y_c).sum(axis=0) / (n - 1)
with np.errstate(divide='ignore', invalid='ignore'):
    rho = cov_xy / (sx * sy)
rho = np.clip(rho, -1.0, 1.0)

# Sigma bruit par colonne
sigma_n = sy * np.sqrt(np.maximum(0.0, 1.0 - rho**2))

# Affichages
print("Colonnes utilisées :", ", ".join(cols))
print("sigma_n (par colonne) :", ", ".join(f"{s:.6g}" for s in sigma_n))
print(f"sigma_n moyen        : {np.nanmean(sigma_n):.6g}")
print(f"sigma_n médian       : {np.nanmedian(sigma_n):.6g}")
