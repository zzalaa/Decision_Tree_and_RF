# Karar Ağaçları ve Rastgele Ormanlar

Sunumdaki ve rapordaki tüm grafikleri ve sayıları üreten koddur.

## VS Code'da çalıştırma 

1. Bu klasörü VS Code'da açın: **File › Open Folder… › `Decision_Tree_and_RF`**
2. Python eklentisi yüklü değilse: sol menü **Extensions** › "Python" (Microsoft) › Install.
3. Terminali açın: **Terminal › New Terminal**.
4. Sanal ortam oluşturup kütüphaneleri kurun:

   **Windows (PowerShell):**
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   PowerShell "scripts is disabled" hatası verirse bir kez şunu çalıştırın:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

   **macOS / Linux:**
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
5. VS Code sağ altta yorumlayıcı sorarsa `.venv` olanı seçin
   (veya **Ctrl+Shift+P › Python: Select Interpreter**).

## Tüm grafikleri üretmek

```
python src/hepsini_calistirir.py
```
Toplam yaklaşık 1–2 dakika sürer (en uzunu s10, OOB yakınsaması).
Grafikler `figures/` klasörüne hem **PNG** (sunum) hem **PDF** (rapor) olarak,
sayısal sonuçlar `results/` klasörüne **JSON** olarak kaydedilir.

Tek bir deneyi çalıştırmak için ilgili dosyayı açıp sağ üstteki ▶ (Run Python File)
düğmesine basabilir ya da terminalde örneğin `python src/s06_bagging_vs_rf.py` yazabilirsiniz.

## Dosyalar ve ödevdeki karşılıkları

| Dosya | Ne yapar |
|---|---|---|
| `src/ortak.py` | Veri seti, grafik stili, kaydetme fonksiyonları | — |
| `s01_bilgi_kazanimi_eslestirme.py` | Bilgi kazanımı ve Gini: elle/kod/scikit-learn eşleştirmesi |
| `s02_entropi_gini.py` | Entropi, Gini ve yanlış sınıflama eğrileri | Kuramsal temeller |
| `s03_karar_sinirlari_derinlik.py` | Derinlik arttıkça parçalanan karar sınırları | Kavramsal kodlama |
| `s04_varyans_yanlilik.py` | Derinliğe göre eğitim/test hatası, RF karşılaştırması | Kuramsal temeller |
| `s05_bootstrap_kararsizlik.py` | Bootstrap ağaçlarının farklı sınırları, ormanın oy haritası | Kavramsal kodlama |
| `s06_bagging_vs_rf.py` | Tek ağaç / bagging / RF sınırları ve ağaç korelasyonu | Kavramsal kodlama |
| `s07_mtry_korelasyon.py` | mtry'ye göre korelasyon ve test hatası | Kavramsal kodlama |
| `s08_varyans_formulu.py` | ρσ² + (1−ρ)σ²/B eğrileri | Kuramsal temeller |
| `s09_oob_olasilik.py` | (1 − 1/n)ⁿ → e⁻¹ ve simülasyonla doğrulama | Kuramsal temeller |
| `s10_oob_yakinsama.py` | B arttıkça OOB hatasının test hatasına yakınsaması | Kavramsal kodlama |
| `s11_mdi_yanlilik.py` | MDI'ın yüksek kardinaliteye yanlılığı, permütasyon önemi | Akademik raporlama |

## Deney verisi

`ortak.deney_verisi()` 8 öznitelikli bir veri üretir: X1, X2 bilgili ("iki ay"),
X3, X4 bunların korelasyonlu kopyaları (r ≈ 0,92–0,96), X5–X8 saf gürültü.
Egitim 400, test 5000 örnektir. Tüm rastgelelik sabit tohumlarla kontrol edilir.
Sonuçlar her çalıştırmada aynıdır (scikit-learn sürümüne göre son basamaklar değişebilir).

Geliştirme ortamı: Python 3, numpy, matplotlib, scikit-learn 1.9.
