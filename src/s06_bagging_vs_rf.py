"""
06 — Tek ağaç vs Bagging vs Random Forest: öznitelik alt-uzay örneklemesinin
karar sınırını nasıl pürüzsüzleştirdiği (ödevin 'kavramsal kodlama' maddesi, 2. kısım).
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from ortak import egitim_test, sinir_ciz, kaydet, sonuc_kaydet, agac_korelasyonu


Xtr, ytr, Xte, yte = egitim_test()
tek = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
bag = RandomForestClassifier(500, max_features=None, random_state=0, n_jobs=-1).fit(Xtr, ytr)   # mtry = p
rf = RandomForestClassifier(500, max_features="sqrt", random_state=0, n_jobs=-1).fit(Xtr, ytr) # mtry = √p

s = {"tek_agac": {"test": tek.score(Xte, yte)},
     "bagging": {"test": bag.score(Xte, yte), "rho": agac_korelasyonu(bag, Xte)},
     "random_forest": {"test": rf.score(Xte, yte), "rho": agac_korelasyonu(rf, Xte)}}
for k, v in s.items():
    print(k, {a: round(b, 3) for a, b in v.items()})

fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
sinir_ciz(axs[0], tek, Xtr, ytr, f"Tek derin ağaç\ntest doğruluğu {s['tek_agac']['test']:.3f}")
sinir_ciz(axs[1], bag, Xtr, ytr, f"Bagging (mtry = p = 8)\ntest {s['bagging']['test']:.3f} · ρ̄ = {s['bagging']['rho']:.2f}")
sinir_ciz(axs[2], rf, Xtr, ytr, f"Random Forest (mtry = √p ≈ 2)\ntest {s['random_forest']['test']:.3f} · ρ̄ = {s['random_forest']['rho']:.2f}")
fig.tight_layout()
kaydet(fig, "s06_bagging_vs_rf")
sonuc_kaydet("s06_bagging_vs_rf", s)
