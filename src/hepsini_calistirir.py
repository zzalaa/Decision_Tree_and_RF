"""Tüm deneyleri sırayla çalıştırır. Kullanım:  python src/hepsini_calistir.py"""
import runpy, time
from pathlib import Path

BETIKLER = ["s01_bilgi_kazanimi_eslestirme", "s02_entropi_gini", "s03_karar_sinirlari_derinlik",
            "s04_varyans_yanlilik", "s05_bootstrap_kararsizlik", "s06_bagging_vs_rf",
            "s07_mtry_korelasyon", "s08_varyans_formulu", "s09_oob_olasilik",
            "s10_oob_yakinsama", "s11_mdi_yanlilik"]
klasor = Path(__file__).resolve().parent
for ad in BETIKLER:
    print(f"\n▶ {ad}.py")
    t = time.time()
    runpy.run_path(str(klasor / f"{ad}.py"), run_name="__main__")
    print(f"  ({time.time() - t:.1f} sn)")
print("\nBitti. Grafikler 'figures/', sayısal sonuçlar 'results/' klasöründe.")
