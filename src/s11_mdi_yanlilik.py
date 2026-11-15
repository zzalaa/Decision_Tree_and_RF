"""
11 — MDI'ın (Mean Decrease Impurity) yüksek kardinaliteli değişkenlere karşı yanlılığı
ve permütasyon önemiyle karşılaştırma (ödevin 'akademik raporlama' maddesi, 2. kısım).

Yalnızca x_bilgili hedefle ilişkilidir. Diğer 5 değişken SAF GÜRÜLTÜDÜR ve yalnızca
kaç farklı değer aldıkları (kardinalite) bakımından farklıdır.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from ortak import kaydet, sonuc_kaydet, YESIL, GRI

rng = np.random.RandomState(42)
n = 1000
x_bilgili = rng.normal(0, 1, n)
y = (x_bilgili + rng.normal(0, 1.2, n) > 0).astype(int)
X = np.column_stack([
    x_bilgili,
    rng.randint(0, 2, n),     # gürültü, 2 farklı değer
    rng.randint(0, 4, n),     # gürültü, 4 farklı değer
    rng.randint(0, 20, n),    # gürültü, 20 farklı değer
    rng.normal(0, 1, n),      # gürültü, sürekli
    rng.permutation(n),       # gürültü, 'ID' (1000 farklı değer)
])
adlar = ["bilgili\n(sürekli)", "gürültü\n2 değer", "gürültü\n4 değer",
         "gürültü\n20 değer", "gürültü\nsürekli", "gürültü\nID"]
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.4, random_state=0)

mdi, perm = [], []
for tohum in range(10):
    orman = RandomForestClassifier(300, random_state=tohum, n_jobs=-1).fit(Xa, ya)
    mdi.append(orman.feature_importances_)                         # eğitim verisinden (in-bag)
    perm.append(permutation_importance(orman, Xb, yb, n_repeats=10,
                                       random_state=tohum).importances_mean)  # test verisinden
mdi, perm = np.array(mdi), np.array(perm)

print(f"{'değişken':<20}{'MDI':>8}{'Permütasyon':>14}")
for a, m, p in zip(adlar, mdi.mean(0), perm.mean(0)):
    print(f"{a.replace(chr(10), ' '):<20}{m:>8.3f}{p:>14.3f}")

renkler = [YESIL] + [GRI] * 5
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].bar(range(6), mdi.mean(0), yerr=mdi.std(0), color=renkler)
axs[0].set_title("MDI (Gini önemi, eğitim verisi)"); axs[0].set_ylabel("Önem")
axs[1].bar(range(6), perm.mean(0), yerr=perm.std(0), color=renkler)
axs[1].axhline(0, color="k", lw=0.7); axs[1].set_title("Permütasyon önemi (test verisi)")
for ax in axs:
    ax.set_xticks(range(6)); ax.set_xticklabels(adlar, fontsize=10)
fig.suptitle("Yalnızca yeşil değişken hedefle ilişkili; gri olanların hepsi saf gürültü", fontsize=13)
fig.tight_layout()
kaydet(fig, "s11_mdi_yanlilik")
sonuc_kaydet("s11_mdi", {"degiskenler": [a.replace("\n", " ") for a in adlar],
                          "mdi_ort": mdi.mean(0).tolist(), "mdi_std": mdi.std(0).tolist(),
                          "perm_ort": perm.mean(0).tolist(), "perm_std": perm.std(0).tolist()})
