"""
10 — Ağaç sayısı (B) arttıkça OOB hatasının genelleme (test) hatasına yakınsaması
(ödevin 'kavramsal kodlama' maddesi, 3. kısım). Bu betik birkaç dakika sürebilir.
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from ortak import egitim_test, kaydet, sonuc_kaydet, MAVI, KIRMIZI

warnings.filterwarnings("ignore")   # küçük B'de 'bazı örnekler OOB değil' uyarısını gizler
Xtr, ytr, Xte, yte = egitim_test()
B_listesi = list(range(5, 30)) + list(range(30, 100, 5)) + list(range(100, 501, 20))
TOHUMLAR = range(5)


def egri(max_features, tohum):
    """warm_start=True ile ağaçları teker teker ekleyip her adımda OOB ve test hatasını ölçer."""
    m = RandomForestClassifier(n_estimators=5, max_features=max_features, oob_score=True,
                               warm_start=True, random_state=tohum, n_jobs=-1)
    oob, test = [], []
    for B in B_listesi:
        m.set_params(n_estimators=B).fit(Xtr, ytr)
        oob.append(1 - m.oob_score_); test.append(1 - m.score(Xte, yte))
    return np.array(oob), np.array(test)


fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
sonuc = {}
for ax, mf, ad in [(axs[0], "sqrt", "Random Forest (mtry = √p)"), (axs[1], None, "Bagging (mtry = p)")]:
    tum_oob, tum_test = [], []
    for t in TOHUMLAR:
        oob, test = egri(mf, t)
        tum_oob.append(oob); tum_test.append(test)
        ax.plot(B_listesi, oob, color=MAVI, alpha=0.3, lw=1)
    ax.plot(B_listesi, np.mean(tum_oob, 0), color=MAVI, lw=2.5, label="OOB hatası (ince: 5 tohum, kalın: ortalama)")
    ax.plot(B_listesi, np.mean(tum_test, 0), color=KIRMIZI, lw=2.5, label="Test hatası (5000 yeni gözlem)")
    ax.set_xscale("log"); ax.set_xlabel("Ağaç sayısı B (log ölçek)"); ax.set_title(ad)
    sonuc[ad] = {"B": B_listesi, "oob_ort": np.mean(tum_oob, 0).tolist(), "test_ort": np.mean(tum_test, 0).tolist()}
    print(f"{ad}:  B=500'de OOB={np.mean(tum_oob,0)[-1]:.3f}  test={np.mean(tum_test,0)[-1]:.3f}")
axs[0].set_ylabel("Hata oranı"); axs[0].legend(frameon=False, fontsize=10)
fig.suptitle("B arttıkça OOB hatası dengelenir ve test (genelleme) hatasına yaklaşır", fontsize=13)
fig.tight_layout()
kaydet(fig, "s10_oob_yakinsama")
sonuc_kaydet("s10_oob_yakinsama", sonuc)
