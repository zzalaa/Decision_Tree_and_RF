"""
01 — Bilgi kazanımının tek bir düğüm için ELLE, KENDİ KODUMUZLA ve SCIKIT-LEARN ile
hesaplanıp birebir eşleştirilmesi (ödevin 'akademik raporlama' maddesi).

Düğüm: 10 örnek (6 pozitif, 4 negatif).
Bölme: sol çocuk 4 örnek (4+, 0-), sağ çocuk 6 örnek (2+, 4-).
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from ortak import sonuc_kaydet

X = np.array([[1], [1], [1], [1], [0], [0], [0], [0], [0], [0]])
y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])


def entropi(etiketler):
    """H(S) = -sum p_k log2 p_k   (p_k = 0 olan sınıflar toplama girmez)"""
    p = np.bincount(etiketler, minlength=2) / len(etiketler)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) + 0.0  # +0.0: "-0.0" yazdırılmasını önler


def gini(etiketler):
    """G(S) = 1 - sum p_k^2"""
    p = np.bincount(etiketler, minlength=2) / len(etiketler)
    return float(1 - (p ** 2).sum())


def kazanim(olcu, ebeveyn, cocuklar):
    """Ebeveyn kirliliği - çocukların örnek sayısıyla ağırlıklı kirliliği"""
    n = len(ebeveyn)
    agirlikli = sum(len(c) / n * olcu(c) for c in cocuklar)
    return olcu(ebeveyn) - agirlikli, agirlikli


sol, sag = y[X[:, 0] == 1], y[X[:, 0] == 0]

print("=== 1) KENDİ KODUMUZLA ===")
ig, ag_h = kazanim(entropi, y, [sol, sag])
dg, ag_g = kazanim(gini, y, [sol, sag])
print(f"H(ebeveyn) = {entropi(y):.5f}   H(sol) = {entropi(sol):.5f}   H(sağ) = {entropi(sag):.5f}")
print(f"Ağırlıklı çocuk entropisi = {ag_h:.5f}   ->  IG = {ig:.5f}")
print(f"G(ebeveyn) = {gini(y):.5f}   G(sol) = {gini(sol):.5f}   G(sağ) = {gini(sag):.5f}")
print(f"Ağırlıklı çocuk Gini'si   = {ag_g:.5f}   ->  ΔG = {dg:.5f}")

print("\n=== 2) SCIKIT-LEARN'ÜN İÇ DEĞERLERİ ===")
sonuc = {"elle_kod": {"H_ebeveyn": entropi(y), "H_sol": entropi(sol), "H_sag": entropi(sag),
                      "IG": ig, "G_ebeveyn": gini(y), "G_sol": gini(sol), "G_sag": gini(sag), "dG": dg}}
for kriter in ("entropy", "gini"):
    agac = DecisionTreeClassifier(criterion=kriter, max_depth=1, random_state=0).fit(X, y).tree_
    imp, n = agac.impurity, agac.n_node_samples
    # Düğüm numaraları: 0 = kök, 1 = sol çocuk (x <= 0.5, yani x = 0 olan 6 örnek), 2 = sağ çocuk
    sk_kazanim = imp[0] - (n[1] / n[0]) * imp[1] - (n[2] / n[0]) * imp[2]
    print(f"[{kriter}] eşik = {agac.threshold[0]:.2f}  impurity = {np.round(imp, 5)}  "
          f"n_node_samples = {n}  ->  kazanım = {sk_kazanim:.5f}")
    sonuc[f"sklearn_{kriter}"] = {"impurity": imp.tolist(), "n_node_samples": n.tolist(),
                                   "kazanim": float(sk_kazanim)}

print("\nNOT: sklearn'de 'sol' çocuk x <= 0.5 koşulunu sağlayan 6 örnektir;")
print("     bizim şemamızdaki 'sağ' çocuğa karşılık gelir. Değerler birebir aynıdır.")
assert np.isclose(sonuc["sklearn_entropy"]["kazanim"], ig)
assert np.isclose(sonuc["sklearn_gini"]["kazanim"], dg)
print("✓ Elle/kod hesabı ile scikit-learn sonuçları eşleşti.")
sonuc_kaydet("s01_bilgi_kazanimi", sonuc)
