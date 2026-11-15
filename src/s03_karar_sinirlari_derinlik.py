"""
03 — Tek bir karar ağacının derinlik arttıkça karar sınırlarında oluşturduğu
yüksek varyanslı parçalanma (ödevin 'kavramsal kodlama' maddesi, 1. kısım).
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from ortak import egitim_test, sinir_ciz, kaydet, sonuc_kaydet

Xtr, ytr, Xte, yte = egitim_test()
print(f"Korelasyon r(X1,X3) = {np.corrcoef(Xtr[:,0], Xtr[:,2])[0,1]:.3f}, "
      f"r(X2,X4) = {np.corrcoef(Xtr[:,1], Xtr[:,3])[0,1]:.3f}")

fig, axs = plt.subplots(1, 4, figsize=(16, 4.3))
sonuc = {}
for ax, d in zip(axs, [1, 3, 6, None]):
    agac = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
    tr, te, yaprak = agac.score(Xtr, ytr), agac.score(Xte, yte), agac.get_n_leaves()
    sonuc[str(d)] = {"egitim_dogruluk": tr, "test_dogruluk": te, "yaprak": int(yaprak)}
    print(f"max_depth={d}:  eğitim={tr:.3f}  test={te:.3f}  yaprak={yaprak}")
    sinir_ciz(ax, agac, Xtr, ytr,
              f"max_depth = {d if d else 'sınırsız'}\neğitim {tr:.2f} · test {te:.2f} · {yaprak} yaprak")
fig.tight_layout()
kaydet(fig, "s03_karar_sinirlari_derinlik")
sonuc_kaydet("s03_derinlik", sonuc)
