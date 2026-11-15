"""05 — Ağaçların kararsızlığı: 6 bootstrap ağacının sınırları ve ormanın oy oranı haritası."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from ortak import egitim_test, kaydet, XX, YY, IZGARA_8D, MAVI, KIRMIZI

Xtr, ytr, Xte, yte = egitim_test()
rng = np.random.RandomState(3)
renkler = plt.cm.tab10(np.arange(6))

fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
for ax in axs:
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c=np.where(ytr == 1, MAVI, KIRMIZI), s=8, edgecolors="none", alpha=0.6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("X1"); ax.set_ylabel("X2")

for i in range(6):
    idx = rng.randint(0, len(ytr), len(ytr))            # yerine koyarak n örnek çek (bootstrap)
    agac = DecisionTreeClassifier(random_state=i).fit(Xtr[idx], ytr[idx])
    Z = agac.predict_proba(IZGARA_8D)[:, 1].reshape(XX.shape)
    axs[0].contour(XX, YY, Z, levels=[0.5], colors=[renkler[i]], linewidths=1.3)
axs[0].set_title("6 bootstrap örnekleminde eğitilmiş 6 derin ağacın\nkarar sınırları (her renk bir ağaç)")

rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1).fit(Xtr, ytr)
Z = rf.predict_proba(IZGARA_8D)[:, 1].reshape(XX.shape)
cs = axs[1].contourf(XX, YY, Z, levels=np.linspace(0, 1, 11), cmap="RdBu", alpha=0.55)
axs[1].contour(XX, YY, Z, levels=[0.5], colors="k", linewidths=1.8)
fig.colorbar(cs, ax=axs[1], fraction=0.04, label="P(sınıf = 1) = ağaçların oy oranı")
axs[1].set_title("500 ağaçlı Random Forest: oyların ortalaması\n(siyah çizgi = %50 sınırı)")
fig.tight_layout()
kaydet(fig, "s05_bootstrap_kararsizlik")
