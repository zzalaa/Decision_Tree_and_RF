"""04 — Derinliğe göre eğitim/test hatası: varyans-yanlılık dengesi ve RF karşılaştırması."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from ortak import egitim_test, kaydet, sonuc_kaydet, MAVI, KIRMIZI, YESIL

Xtr, ytr, Xte, yte = egitim_test()
derinlikler = list(range(1, 21))
egitim_hata, test_hata = [], []
for d in derinlikler:
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
    egitim_hata.append(1 - m.score(Xtr, ytr))
    test_hata.append(1 - m.score(Xte, yte))

rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1).fit(Xtr, ytr)
rf_hata = 1 - rf.score(Xte, yte)
en_iyi = int(np.argmin(test_hata))
print(f"En iyi tek ağaç derinliği = {derinlikler[en_iyi]}, test hatası = {test_hata[en_iyi]:.3f}")
print(f"Random Forest (500 ağaç, budanmamış) test hatası = {rf_hata:.3f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(derinlikler, egitim_hata, "o-", color=MAVI, ms=4, label="Tek ağaç — eğitim hatası")
ax.plot(derinlikler, test_hata, "o-", color=KIRMIZI, ms=4, label="Tek ağaç — test hatası")
ax.axhline(rf_hata, color=YESIL, ls="--", lw=2, label=f"Random Forest — test hatası ({rf_hata:.3f})")
ax.annotate("en iyi derinlik", (derinlikler[en_iyi], test_hata[en_iyi]),
            (derinlikler[en_iyi] + 3, test_hata[en_iyi] + 0.06), arrowprops=dict(arrowstyle="->"))
ax.text(1.3, max(test_hata) * 0.93, "yüksek yanlılık\n(underfitting)", fontsize=10)
ax.text(14, min(egitim_hata) + 0.025, "yüksek varyans\n(overfitting)", fontsize=10)
ax.set_xlabel("Ağaç derinliği (max_depth)")
ax.set_ylabel("Hata oranı")
ax.set_title("Derinlik arttıkça eğitim hatası düşer, test hatası 'U' çizer")
ax.legend(frameon=False)
kaydet(fig, "s04_varyans_yanlilik")
sonuc_kaydet("s04_varyans_yanlilik", {"derinlik": derinlikler, "egitim_hata": egitim_hata,
                                      "test_hata": test_hata, "rf_test_hata": rf_hata})
