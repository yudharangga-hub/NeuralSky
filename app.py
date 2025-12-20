import os
import time
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, url_for, redirect, send_file
import numpy as np
import cv2
import requests  # Library untuk akses data BMKG
import urllib3

# Nonaktifkan peringatan SSL untuk request ke BMKG (agar lancar di local)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- IMPORT MODUL CUSTOM ---
# Pastikan file gradcam_pytorch.py, impact_logic.py, report_generator.py ada di folder yang sama
from gradcam_pytorch import GradCAM, save_gradcam 

try:
    from impact_logic import get_weather_impact
    from report_generator import create_pdf
    REPORT_FEATURE_ACTIVE = True
except ImportError:
    REPORT_FEATURE_ACTIVE = False
    print("[WARN] Modul Impact/Report tidak ditemukan.")

app = Flask(__name__)

# --- KONFIGURASI FOLDER ---
UPLOAD_FOLDER = 'static/uploads'
CAM_FOLDER = 'static/heatmaps'
MODEL_FOLDER = 'models_pytorch' 
TRAIN_DIR = 'dataset/clouds_train'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CAM_FOLDER'] = CAM_FOLDER
app.config['TRAIN_DIR'] = TRAIN_DIR

# Buat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CAM_FOLDER, exist_ok=True)

# --- SETUP DEVICE (GPU/CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Server berjalan di: {device}")

# --- DAFTAR KELAS & WILAYAH BMKG ---
if os.path.exists(TRAIN_DIR):
    CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))
else:
    CLASS_NAMES = ['cirriform clouds', 'clear sky', 'cumulonimbus clouds', 
                   'cumulus clouds', 'high cumuliform clouds', 
                   'stratiform clouds', 'stratocumulus clouds']

NUM_CLASSES = len(CLASS_NAMES)

# Daftar Kode Wilayah untuk Dropdown
# Referensi: Kepmendagri / Database BMKG
WILAYAH_DICT = {
    "36.74.06.1005": "PAMULANG (TANGSEL)",        # Default Project
    "31.71.01.1001": "GAMBIR (JAKARTA PUSAT)",    # Pusat Jakarta
    "36.74.07.1002": "PAKUALAM (SERPONG UTARA)",  # GANTI BANDARA -> PAKUALAM
    "32.71.01.1001": "BOGOR TENGAH",              # Data Hujan Kota Hujan
    "51.03.01.1001": "KUTA (BALI)",               # Wisata
    "34.71.11.1001": "KRATON (YOGYAKARTA)"        # Daerah Istimewa
}

# --- DEFINISI MODEL (SIMPLE CNN) ---
# Harus didefinisikan ulang agar state_dict bisa dimuat
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- LOAD MODELS ---
models_dict = {}
def load_models():
    print("[INFO] Loading PyTorch Models...")
    # 1. Simple CNN
    try:
        model = SimpleCNN(NUM_CLASSES)
        path = os.path.join(MODEL_FOLDER, 'simple_cnn_v2.pth')
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device).eval()
            models_dict['Simple CNN'] = model
    except Exception as e: print(f"[ERR] Load SimpleCNN: {e}")

    # 2. MobileNetV2
    try:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
        path = os.path.join(MODEL_FOLDER, 'mobilenet_finetuned.pth')
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device).eval()
            models_dict['MobileNetV2'] = model
    except Exception as e: print(f"[ERR] Load MobileNet: {e}")

    # 3. EfficientNetB0
    try:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
        path = os.path.join(MODEL_FOLDER, 'efficientnet_finetuned.pth')
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device).eval()
            models_dict['EfficientNetB0'] = model
    except Exception as e: print(f"[ERR] Load EfficientNet: {e}")
load_models()

# --- HELPER IMAGE TRANSFORM ---
val_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- FUNGSI CV: OKTAS (CLOUD COVERAGE) ---
def analyze_cloud_properties(filepath):
    try:
        img = cv2.imread(filepath)
        if img is None: return 0, 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Thresholding: Memisahkan awan (putih) dan langit (gelap)
        _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        total_pixels = img.shape[0] * img.shape[1]
        cloud_pixels = cv2.countNonZero(thresh)
        coverage_percent = (cloud_pixels / total_pixels) * 100
        oktas = round((coverage_percent / 100) * 8)
        return round(coverage_percent, 1), oktas
    except Exception as e:
        print(f"[ERR] Gagal hitung oktas: {e}")
        return 0, 0

# --- FUNGSI BMKG: FETCH DATA DINAMIS ---
def get_bmkg_data(kode_wilayah):
    api_url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={kode_wilayah}"
    
    # Headers User-Agent agar tidak diblokir server BMKG
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"[INFO] Mengambil data BMKG: {api_url}")
        response = requests.get(api_url, headers=headers, timeout=10, verify=False)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[ERR] Status Code BMKG: {response.status_code}")
    except Exception as e:
        print(f"[ERR] Exception Request BMKG: {e}")
    return None

def get_satellite_image():
    # URL Statis Citra Satelit Himawari-9 (IR Enhanced)
    return "https://inderaja.bmkg.go.id/IMAGE/HIMA/H08_EH_Indonesia.png"

# --- ROUTE UTAMA (DASHBOARD) ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        if file.filename == '': return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('index', filename=filename))

    filename = request.args.get('filename')
    
    # Inisialisasi variabel agar tidak error di Jinja2
    results = []
    final_decision = None
    heatmap_data = {}
    impact_info = None
    image_url = None
    evidence_files = []
    cloud_coverage = 0
    oktas_val = 0

    if filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            image_url = url_for('static', filename=f'uploads/{filename}')
            
            # 1. Hitung Oktas
            cloud_coverage, oktas_val = analyze_cloud_properties(filepath)

            # 2. AI Inference
            img_pil = Image.open(filepath).convert('RGB')
            img_tensor = val_transform(img_pil).unsqueeze(0).to(device)
            vote_box = {}

            for name, model in models_dict.items():
                try:
                    start_time = time.time()
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                    score, idx = torch.max(probabilities, 1)
                    idx = idx.item()
                    score = score.item()
                    prediction = CLASS_NAMES[idx]
                    
                    results.append({
                        'model': name, 'prediction': prediction,
                        'confidence': round(score * 100, 2),
                        'latency': round((time.time() - start_time) * 1000, 2)
                    })

                    if prediction not in vote_box: vote_box[prediction] = []
                    vote_box[prediction].append(score)

                    # 3. Grad-CAM Logic
                    target_layer = None
                    if name == 'Simple CNN': target_layer = model.features[-2]
                    elif name == 'MobileNetV2': target_layer = model.features[-1] 
                    elif name == 'EfficientNetB0': target_layer = model.features[-1]

                    if target_layer:
                        cam_extractor = GradCAM(model, target_layer)
                        heatmap = cam_extractor(img_tensor, class_idx=idx)
                        cam_path = os.path.join(app.config['CAM_FOLDER'], f"cam_{name}_{filename}")
                        save_gradcam(filepath, heatmap, cam_path)
                        heatmap_data[name] = {'img': url_for('static', filename=f'heatmaps/cam_{name}_{filename}'), 
                                              'conf': round(score*100, 2), 'pred': prediction}
                except Exception as e: print(f"[ERR] Inference {name}: {e}")

            # 4. Ensemble Weighted Voting
            best_class = None
            highest_score = -1
            for cls, scores in vote_box.items():
                weighted = (sum(scores)/len(scores)) + (len(scores) * 0.15)
                if weighted > highest_score:
                    highest_score = weighted
                    best_class = cls
            
            if best_class:
                final_decision = {'class': best_class, 'confidence': round((sum(vote_box[best_class])/len(vote_box[best_class])) * 100, 2)}
                
                # 5. Sistem Pakar & Evidence
                if REPORT_FEATURE_ACTIVE:
                    impact_info = get_weather_impact(best_class)
                    try:
                        class_path = os.path.join(app.config['TRAIN_DIR'], best_class)
                        if os.path.exists(class_path):
                            all_imgs = [i for i in os.listdir(class_path) if i.lower().endswith(('.png','.jpg'))]
                            evidence_files = [f"{best_class}/{i}" for i in random.sample(all_imgs, min(len(all_imgs), 3))]
                    except: pass

    return render_template('index.html',
                           results=results, image_file=image_url,
                           final_decision=final_decision, impact_info=impact_info,
                           heatmap_data=heatmap_data, evidence_files=evidence_files,
                           class_names=CLASS_NAMES, report_active=REPORT_FEATURE_ACTIVE,
                           cloud_coverage=cloud_coverage, oktas_val=oktas_val)

# --- ROUTE UTILS ---
@app.route('/dataset_image/<path:filename>')
def dataset_image(filename):
    return send_file(os.path.join(app.config['TRAIN_DIR'], filename))

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if not REPORT_FEATURE_ACTIVE: return "Feature Disabled", 500
    filename = request.form.get('filename')
    corrected_class = request.form.get('corrected_class')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    confidence_str = "MANUAL VALIDATION"
    coverage, oktas = analyze_cloud_properties(filepath)
    impacts = get_weather_impact(corrected_class)
    
    heatmap_path = f"static/heatmaps/cam_EfficientNetB0_{filename}"
    if not os.path.exists(heatmap_path): heatmap_path = filepath
    
    pdf_path = create_pdf(filename, corrected_class, confidence_str, impacts, heatmap_path, coverage, oktas)
    return send_file(pdf_path, as_attachment=True, download_name=f"Laporan_{corrected_class}_{filename.split('.')[0]}.pdf")

# --- ROUTE DATA BMKG (UPDATED WITH DROPDOWN LOGIC) ---
@app.route('/bmkg_feed')
def bmkg_feed():
    try:
        # Ambil kode wilayah dari parameter URL, default ke Pamulang jika kosong
        kode = request.args.get('kode', "36.74.06.1005")
        
        weather_data = get_bmkg_data(kode)
        satellite_url = get_satellite_image()
        
        lokasi_info = {}
        flat_forecast = []
        
        if weather_data and 'lokasi' in weather_data:
            lokasi_info = weather_data['lokasi']
            
            # Logic parsing JSON BMKG (Flattening Nested List)
            if 'data' in weather_data and weather_data['data']:
                # Data BMKG strukturnya: data -> [0] -> cuaca -> [list hari] -> [list jam]
                cuaca_per_hari = weather_data['data'][0]['cuaca']
                
                # Loop setiap hari, lalu loop setiap jam
                for hari in cuaca_per_hari:
                    for jam in hari:
                        flat_forecast.append(jam)
        
        # Render template dengan data dinamis
        return render_template('components/tab_bmkg.html', 
                               lokasi=lokasi_info, 
                               forecasts=flat_forecast,
                               satellite=satellite_url,
                               daftar_wilayah=WILAYAH_DICT, # Kirim daftar wilayah ke HTML
                               current_kode=kode)           # Kirim kode yang sedang aktif
    except Exception as e:
        print(f"[ERR] Route bmkg_feed error: {e}")
        return f"<div class='text-danger p-5'>SYSTEM ERROR: {e}</div>", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)