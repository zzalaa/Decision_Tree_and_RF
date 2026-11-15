"""
Ortak yardımcılar: klasör yolları, grafik stili, deney veri seti ve kaydetme fonksiyonları.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons

# --- Klasörler ---
KOK = Path(__file__).resolve().parent.parent       # proje kök klasörü
FIG = KOK / "figures"                              # grafikle
SONUC = KOK / "results"                            # sayısal sonuçlar
FIG.mkdir(exist_ok=True)
SONUC.mkdir(exist_ok=True)

# --- Grafik stili ---
MAVI, KIRMIZI, YESIL, MOR, TURUNCU, GRI = "#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#ea580c", "#94a3b8"
ARKA_CMAP = ListedColormap(["#fde2e2", "#dbeafe"])   # sınıf 0 = açık kırmızı, sınıf 1 = açık mavi

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # Türkçe karakterleri destekler
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

RASTGELE_TOHUM = 0  # Tekrarlanabilirlik için sabit tohum


def kaydet(fig, ad):
    for uzanti in ("png", "pdf"):
        fig.savefig(FIG / f"{ad}.{uzanti}")
    plt.close(fig)
    print(f"  ✓ figures/{ad}.png ve .pdf kaydedildi")


def sonuc_kaydet(ad, sozluk):
    with open(SONUC / f"{ad}.json", "w", encoding="utf-8") as f:
        json.dump(sozluk, f, ensure_ascii=False, indent=2, default=float)
    print(f"  ✓ results/{ad}.json kaydedildi")


# --- Deney veri seti ---
def deney_verisi(n, tohum):
    """
    Kontrollü korelasyonlu ve gürültülü özniteliklerden oluşan veri seti (p = 8).

    X1, X2 : 'iki ay' biçimindeki BİLGİLİ öznitelikler (etiket gürültüsü 0.30)
    X3, X4 : X1 ve X2'nin KORELASYONLU kopyaları (X + N(0, 0.25^2))
    X5..X8 : hedefle ilgisi olmayan SAF GÜRÜLTÜ, N(0, 1)
    """
    rng = np.random.RandomState(tohum)
    X2d, y = make_moons(n, noise=0.30, random_state=tohum)
    x3 = X2d[:, 0] + rng.normal(0, 0.25, n)
    x4 = X2d[:, 1] + rng.normal(0, 0.25, n)
    gurultu = rng.normal(0, 1, (n, 4))
    X = np.column_stack([X2d, x3, x4, gurultu])
    return X, y


def egitim_test():
    """Tüm deneylerde aynı eğitim (400) ve test (5000) setleri kullanılır."""
    Xtr, ytr = deney_verisi(400, 0)
    Xte, yte = deney_verisi(5000, 1)
    return Xtr, ytr, Xte, yte


# --- Karar sınırı çizimi ---
# Sınırı (X1, X2) düzleminde çiziyoruz. Diğer öznitelikleri sabitlememiz gerekir:
# korelasyonlu kopyalar X3 = X1, X4 = X2; gürültü öznitelikleri ortalaması olan 0.
XX, YY = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.5, 2, 300))
_IZGARA_2D = np.c_[XX.ravel(), YY.ravel()]
IZGARA_8D = np.column_stack([_IZGARA_2D, _IZGARA_2D[:, 0], _IZGARA_2D[:, 1],
                             np.zeros((len(_IZGARA_2D), 4))])


def sinir_ciz(ax, model, X, y, baslik):
    """Bir modelin karar bölgelerini ve sınırını eğitim noktalarıyla birlikte çizer."""
    Z = model.predict_proba(IZGARA_8D)[:, 1].reshape(XX.shape)
    ax.contourf(XX, YY, Z, levels=[0, 0.5, 1], cmap=ARKA_CMAP, alpha=0.9)
    ax.contour(XX, YY, Z, levels=[0.5], colors="k", linewidths=1.2)
    ax.scatter(X[:, 0], X[:, 1], c=np.where(y == 1, MAVI, KIRMIZI), s=9, edgecolors="none")
    ax.set_title(baslik, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("X1"); ax.set_ylabel("X2")


# --- Ormanlar için yardımcı ---
def agac_korelasyonu(orman, X, k=100):
    """İlk k ağacın olasılık tahminleri arasındaki ortalama ikili korelasyon (ρ̄)."""
    P = np.array([agac.predict_proba(X)[:, 1] for agac in orman.estimators_[:k]])
    C = np.corrcoef(P)
    return float(C[np.triu_indices(len(C), 1)].mean())
