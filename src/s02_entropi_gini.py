"""02 — İkili sınıflamada entropi, Gini ve yanlış sınıflama oranının karşılaştırılması."""
import numpy as np
import matplotlib.pyplot as plt
from ortak import kaydet, MAVI, KIRMIZI

p = np.linspace(1e-4, 1 - 1e-4, 400)
H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
G = 1 - (p ** 2 + (1 - p) ** 2)
yanlis = 1 - np.maximum(p, 1 - p)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(p, H, color=MAVI, lw=2.5, label="Entropi  H(p)")
ax.plot(p, G, color=KIRMIZI, lw=2.5, label="Gini  G(p)")
ax.plot(p, 2 * G, color=KIRMIZI, lw=1.4, ls="--", label="2 × Gini (ölçek karşılaştırması)")
ax.plot(p, yanlis, color="gray", lw=1.4, ls=":", label="Yanlış sınıflama oranı")
ax.axvline(0.5, color="k", lw=0.6, alpha=0.4)
ax.set_xlabel("p = düğümdeki pozitif sınıf oranı")
ax.set_ylabel("Kirlilik")
ax.set_title("İkili sınıflamada kirlilik ölçüleri")
ax.legend(frameon=False)
kaydet(fig, "s02_entropi_gini")
