"""09 — Bir örneğin bootstrap örnekleminde hiç seçilmeme olasılığı: (1 - 1/n)^n → e^-1 ≈ 0.368."""
import numpy as np
import matplotlib.pyplot as plt
from ortak import kaydet, sonuc_kaydet, MAVI, KIRMIZI

n = np.arange(1, 101)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(n, (1 - 1 / n) ** n, "o", color=MAVI, ms=4, label="(1 − 1/n)ⁿ")
ax.axhline(np.exp(-1), color=KIRMIZI, ls="--", lw=2, label="e⁻¹ ≈ 0,368")
ax.set_xlabel("Veri seti büyüklüğü n"); ax.set_ylabel("Seçilmeme olasılığı"); ax.set_ylim(0, 0.42)
ax.set_title("Bir bootstrap örnekleminin dışında kalan veri oranı")
ax.legend(frameon=False)
kaydet(fig, "s09_oob_olasilik")

# Simülasyonla doğrulama: n = 1000, 500 bootstrap örneklemi
rng = np.random.RandomState(0)
oranlar = [len(np.setdiff1d(np.arange(1000), rng.randint(0, 1000, 1000))) / 1000 for _ in range(500)]
print(f"Simülasyon: dışarıda kalan ortalama oran = {np.mean(oranlar):.4f}  (kuram: {np.exp(-1):.4f})")
sonuc_kaydet("s09_oob_olasilik", {"simulasyon": float(np.mean(oranlar)), "kuram": float(np.exp(-1))})
