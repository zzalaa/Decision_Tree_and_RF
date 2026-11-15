"""08 — Korelasyonlu ağaçların ortalamasının varyansı: Var = ρσ² + (1-ρ)σ²/B  (σ² = 1)."""
import numpy as np
import matplotlib.pyplot as plt
from ortak import kaydet, YESIL, MAVI, MOR, KIRMIZI

B = np.arange(1, 201)
fig, ax = plt.subplots(figsize=(8, 4.5))
for rho, renk in zip([0, 0.1, 0.3, 0.6, 1.0], ["k", YESIL, MAVI, MOR, KIRMIZI]):
    ax.plot(B, rho + (1 - rho) / B, color=renk, lw=2.5, label=f"ρ = {rho}")
    ax.axhline(rho, color=renk, lw=0.7, ls=":")      # B → ∞ iken inilebilecek taban (ρσ²)
ax.set_xscale("log")
ax.set_xlabel("Ağaç sayısı B (log ölçek)")
ax.set_ylabel("Var(ortalama) / σ²")
ax.set_title("Ortalamanın varyansı: ρσ² + (1 − ρ)σ² / B")
ax.legend(frameon=False, ncol=2)
kaydet(fig, "s08_varyans_formulu")
