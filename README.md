# ☁️ NEURAL SKY: Cloud Classification & Weather Analysis System

> **Sistem Klasifikasi Awan Berbasis Hybrid Ensemble Deep Learning dengan Integrasi Explainable AI (XAI) dan Sistem Pakar Meteorologi.**

![Main Dashboard](screenshots/dashboard_main.jpg)

## 📖 Tentang Proyek
**Neural Sky** adalah purwarupa sistem cerdas (*Intelligent Decision Support System*) yang dirancang untuk membantu pengamatan meteorologi dan keselamatan penerbangan. Sistem ini tidak hanya mengklasifikasikan jenis awan, tetapi juga memberikan estimasi kuantitatif (Oktas), transparansi visual (XAI), dan simulasi dampak cuaca.

Proyek ini dikembangkan sebagai bagian dari Tesis Magister Teknik Informatika (Universitas Pamulang) untuk menjawab tantangan subjektivitas dalam pengamatan awan manual.

### 🌟 Fitur Unggulan
1.  **Hybrid Ensemble Learning:** Menggabungkan 3 arsitektur CNN (*EfficientNet-B0, MobileNetV2, Simple CNN*) dengan mekanisme *Weighted Voting* untuk akurasi maksimal.
2.  **Neural Vision (XAI):** Visualisasi *Grad-CAM* untuk membuka "kotak hitam" AI, menampilkan area fokus model pada citra awan.
3.  **Oktas Estimation:** Perhitungan otomatis tutupan awan (0-8 skala Oktas) menggunakan algoritma *Computer Vision* (Thresholding).
4.  **Forensic Evidence:** Fitur *Content-Based Retrieval* yang menampilkan citra referensi dari dataset untuk memvalidasi kemiripan pola.
5.  **Weather Simulation (T+1):** Sistem Pakar (*Rule-Based*) yang memproyeksikan potensi cuaca 1 jam ke depan berdasarkan morfologi awan.
6.  **Human-in-the-Loop:** Fitur validasi manual yang memungkinkan pakar mengoreksi prediksi sebelum mencetak laporan PDF.
7.  **Voice Tactical Briefing:** Asisten suara otomatis yang membacakan hasil analisis layaknya *copilot*.

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
* **Bahasa:** Python 3.10
* **Deep Learning Framework:** PyTorch (Torchvision)
* **Web Framework:** Flask (Jinja2 Template)
* **Computer Vision:** OpenCV (cv2)
* **Data Visualization:** Matplotlib, Seaborn
* **Frontend:** Bootstrap 5, CSS3 (HUD Style), JavaScript
* **Report Gen:** FPDF

---

## 📂 Struktur Folder
```text
Neural-Sky/
├── app.py                  # Server Utama (Flask)
├── gradcam_pytorch.py      # Engine XAI (Grad-CAM)
├── impact_logic.py         # Knowledge Base Sistem Pakar
├── train_pytorch.py        # Script Pelatihan Model
├── dataset/                # Dataset Citra Awan (Train/Test)
├── models_pytorch/         # File Model Terlatih (.pth)
├── static/                 
│   ├── css/                # Style HUD & UI
│   ├── reports/            # Output Laporan PDF
│   └── training_history/   # Grafik Evaluasi Model
└── templates/              # Antarmuka Pengguna (HTML Modular)

```
---

## 🚀 Cara Instalasi & Menjalankan
1. Clone Repository
Bash

git clone [https://github.com/yudharangga-hub/NeuralSky.git](https://github.com/yudharangga-hub/NeuralSky.git)
cd NeuralSky
2. Setup Environment
Disarankan menggunakan Virtual Environment (Python 3.10+):

Bash

python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
3. Install Dependencies
Bash

pip install -r requirements.txt
(Buat file requirements.txt berisi: flask, torch, torchvision, opencv-python, fpdf, matplotlib, seaborn, pillow)

4. Siapkan Dataset & Model
Pastikan folder dataset/clouds_train berisi 7 sub-folder kelas awan.

Pastikan file model .pth sudah ada di folder models_pytorch/ (atau jalankan python train_pytorch.py untuk melatih dari awal).

5. Jalankan Aplikasi
Bash

python app.py
Buka browser dan akses: http://127.0.0.1:5000

---

## 🛡️ Disclaimer
Fitur Simulasi Cuaca (T+1) pada aplikasi ini disusun menggunakan pendekatan Expert System berdasarkan literatur meteorologi. Hasil proyeksi bersifat teoritis dan tidak menggantikan prakiraan cuaca numerik (NWP) resmi dari instansi terkait yang menggunakan data sensor multi-variabel.

## 👨‍💻 Author
Yudha Rangga Wulung Pura Mahasiswa Magister Teknik Informatika - Universitas Pamulang

Research Interest: Computer Vision, Deep Learning, Meteorology.

Copyright © 2025 Neural Sky Project. All Rights Reserved.