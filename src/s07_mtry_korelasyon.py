"""07 — mtry'ye göre ağaçlar arası korelasyon ve test hatası (güç-korelasyon dengesi)."""
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from ortak import egitim_test, kaydet, sonuc_kaydet, agac_korelasyonu, MOR, KIRMIZI

Xtr, ytr, Xte, yte = egitim_test()
mtry_degerleri = [1, 2, 3, 4, 6, 8]
rho, hata = [], []
for m in mtry_degerleri:
    orman = RandomForestClassifier(300, max_features=m, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    rho.append(agac_korelasyonu(orman, Xte)); hata.append(1 - orman.score(Xte, yte))
    print(f"mtry={m}:  ρ̄={rho[-1]:.3f}  test hatası={hata[-1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax.plot(mtry_degerleri, rho, "o-", color=MOR, lw=2, label="Ağaçlar arası ortalama korelasyon ρ̄")
ax2.plot(mtry_degerleri, hata, "s--", color=KIRMIZI, lw=2, label="Test hatası")
ax.set_xlabel("mtry (her düğümde denenen öznitelik sayısı, p = 8)")
ax.set_ylabel("ρ̄", color=MOR); ax2.set_ylabel("Test hatası", color=KIRMIZI)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
ax.set_title("mtry küçüldükçe ağaçlar birbirinden bağımsızlaşır")
kaydet(fig, "s07_mtry_korelasyon")
sonuc_kaydet("s07_mtry", {"mtry": mtry_degerleri, "rho": rho, "test_hata": hata})
