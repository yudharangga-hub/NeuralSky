# ☁️ NEURAL SKY: Cloud Classification & Weather Analysis System

> **Sistem Klasifikasi Awan Berbasis Hybrid Ensemble Deep Learning dengan Integrasi Explainable AI (XAI) dan Sistem Pakar Meteorologi.**

![Main Dashboard](screenshots/dashboard_main.jpg)

## 📖 Tentang Proyek

**Neural Sky** adalah purwarupa sistem cerdas (*Intelligent Decision Support System*) yang dirancang untuk membantu pengamatan meteorologi dan keselamatan penerbangan. Sistem ini tidak hanya mengklasifikasikan jenis awan, tetapi juga memberikan estimasi kuantitatif (Oktas), transparansi visual (XAI), dan simulasi dampak cuaca.

Proyek ini dikembangkan sebagai bagian dari **Tesis Magister Teknik Informatika (Universitas Pamulang)** untuk menjawab tantangan subjektivitas dalam pengamatan awan manual.

### 🌟 Fitur Unggulan

1.  **Hybrid Ensemble Learning:** Menggabungkan 3 arsitektur CNN (*EfficientNet-B0, MobileNetV2, Simple CNN*) dengan mekanisme *Weighted Voting* untuk akurasi maksimal.
2.  **Neural Vision (XAI):** Visualisasi *Grad-CAM* untuk membuka "kotak hitam" AI, menampilkan area fokus model pada citra awan.
3.  **Oktas Estimation:** Perhitungan otomatis tutupan awan (0-8 skala Oktas) menggunakan algoritma *Computer Vision* (Thresholding).
4.  **Forensic Evidence:** Fitur *Content-Based Retrieval* yang menampilkan citra referensi dari dataset untuk memvalidasi kemiripan pola.
5.  **Weather Simulation (T+1):** Sistem Pakar (*Rule-Based*) yang memproyeksikan potensi cuaca 1 jam ke depan berdasarkan morfologi awan.
6.  **BMKG Station Integration:** Integrasi data stasiun meteorologi BMKG untuk informasi lokasi dan kondisi cuaca.
7.  **Human-in-the-Loop:** Fitur validasi manual yang memungkinkan pakar mengoreksi prediksi sebelum mencetak laporan PDF.
8.  **Voice Tactical Briefing:** Asisten suara otomatis yang membacakan hasil analisis layaknya *copilot*.

---

## 📸 Galeri Fitur

| **Target Scan (Klasifikasi & Oktas)** | **Neural Vision (Explainable AI)** |
|:---:|:---:|
| ![Scan](screenshots/feature_scan.jpg) | ![XAI](screenshots/feature_xai.jpg) |
| *Deteksi jenis awan, estimasi oktas, dan analisis dampak.* | *Heatmap Grad-CAM untuk transparansi keputusan AI.* |

| **Forensic Evidence (Database Match)** | **Lab Metrics (Performance)** |
|:---:|:---:|
| ![Forensic](screenshots/feature_forensic.jpg) | ![Metrics](screenshots/feature_metrics.jpg) |
| *Pencocokan pola visual dengan dataset referensi.* | *Evaluasi model (Confusion Matrix & Learning Curve).* |

---

## 🛠️ Teknologi yang Digunakan

| Kategori | Teknologi |
|----------|-----------|
| **Bahasa** | Python 3.10+ |
| **Deep Learning** | PyTorch (Torchvision) |
| **Web Framework** | Flask (Jinja2 Template) |
| **Computer Vision** | OpenCV (cv2) |
| **Data Visualization** | Matplotlib, Seaborn |
| **Frontend** | Bootstrap 5, CSS3 (HUD Style), JavaScript |
| **Report Generation** | FPDF |

---

## 📂 Struktur Folder

```
NeuralSky/
├── app.py                      # Server utama Flask
├── train_pytorch.py            # Script pelatihan model CNN
├── evaluate_pytorch.py         # Script evaluasi model
├── gradcam_pytorch.py          # Engine XAI (Grad-CAM)
├── impact_logic.py             # Knowledge base sistem pakar
├── report_generator.py         # Generator laporan PDF
├── generate_oktas_proof.py     # Generator bukti visual Oktas
├── generate_comparison.py      # Generator perbandingan model
├── reconstruct_all_charts.py   # Rekonstruksi semua grafik
├── requirements.txt            # Daftar dependensi Python
├── README.md                   # Dokumentasi proyek
│
├── dataset/                    # Dataset citra awan
│   ├── clouds_train/           # Data pelatihan (7 kelas)
│   │   ├── cirriform clouds/
│   │   ├── clear sky/
│   │   ├── cumulonimbus clouds/
│   │   ├── cumulus clouds/
│   │   ├── high cumuliform clouds/
│   │   ├── stratiform clouds/
│   │   └── stratocumulus clouds/
│   └── clouds_test/            # Data pengujian (7 kelas)
│
├── models_pytorch/             # Model terlatih (tidak di-commit ke Git)
│   ├── efficientnet_finetuned.pth
│   ├── mobilenet_finetuned.pth
│   └── simple_cnn_v2.pth
│
├── static/
│   ├── css/
│   │   └── hud_style.css       # Styling HUD interface
│   ├── data/
│   │   ├── bmkg_stations.json  # Data stasiun BMKG
│   │   └── database/           # Database aplikasi
│   │       ├── analyses.csv
│   │       ├── evidence.csv
│   │       ├── gradcam.csv
│   │       ├── predictions.csv
│   │       ├── upload_history.csv
│   │       └── uploads.csv
│   ├── heatmaps/               # Output visualisasi Grad-CAM
│   ├── images/                 # Logo dan aset statis
│   ├── reports/                # Laporan PDF yang dihasilkan
│   ├── training_history/       # Grafik history pelatihan
│   │   ├── efficientnet_chart.png
│   │   ├── mobilenet_chart.png
│   │   ├── simple_cnn_chart.png
│   │   ├── model_comparison.png
│   │   └── oktas_proof.png
│   └── uploads/                # Citra yang diupload pengguna
│
├── templates/                  # Template HTML
│   ├── index.html              # Halaman utama
│   └── components/             # Komponen modular
│       ├── css_styles.html
│       ├── scripts.html
│       ├── tab_bmkg.html
│       ├── tab_evidence.html
│       ├── tab_metrics.html
│       ├── tab_scan.html
│       ├── tab_simulation.html
│       └── tab_xray.html
│
└── screenshots/                # Screenshot untuk dokumentasi
    ├── dashboard_main.jpg
    ├── feature_scan.jpg
    ├── feature_xai.jpg
    ├── feature_forensic.jpg
    └── feature_metrics.jpg
```

---

## 🚀 Cara Instalasi & Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/yudharangga-hub/NeuralSky.git
cd NeuralSky
```

### 2. Setup Environment

Disarankan menggunakan Virtual Environment (Python 3.10+):

```bash
python -m venv venv
venv\Scripts\activate     # Windows
# atau: source venv/bin/activate  # Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Siapkan Dataset & Model

- Pastikan folder `dataset/clouds_train` berisi 7 sub-folder kelas awan
- Pastikan file model `.pth` ada di folder `models_pytorch/`
- Atau jalankan `python train_pytorch.py` untuk melatih dari awal

### 5. Jalankan Aplikasi

```bash
python app.py
```

Buka browser dan akses: **http://127.0.0.1:5000**

---

## 📊 Kelas Awan yang Didukung

| Kode | Nama Kelas Awan | Deskripsi |
|------|-----------------|-----------|
| CC | Cirriform Clouds | Awan berbentuk fiber, tinggi, tipis |
| CS | Clear Sky | Langit cerah tanpa awan |
| CB | Cumulonimbus Clouds | Awan badai, berpotensi hujan lebat |
| CU | Cumulus Clouds | Awan bergumpal, biasanya baik |
| HC | High Cumuliform Clouds | Awan cumuliform tinggi |
| ST | Stratiform Clouds | Awan berlapis, mendung |
| SC | Stratocumulus Clouds | Awan gumpal berlapis |

---

## 🛡️ Disclaimer

Fitur Simulasi Cuaca (T+1) pada aplikasi ini disusun menggunakan pendekatan Expert System berdasarkan literatur meteorologi. Hasil proyeksi bersifat teoritis dan tidak menggantikan prakiraan cuaca numerik (NWP) resmi dari instansi terkait yang menggunakan data sensor multi-variabel.

---

## 📝 Lisensi

Proyek ini dikembangkan untuk keperluan akademik (Tesis Magister).