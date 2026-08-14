import os
import time
import random
import uuid
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, url_for, redirect, send_file, session
import numpy as np
import cv2
import requests  # Library untuk akses data BMKG
import json
import urllib3
import csv
from datetime import datetime, timezone
from datetime import timedelta
from pathlib import Path
from functools import wraps

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
app.secret_key = os.environ.get('SECRET_KEY', 'neural_sky_demo_secret_2026')

USERS = {
    'admin': {'password': 'admin', 'role': 'admin'},
    'user': {'password': 'user', 'role': 'user'}
}

# --- KONFIGURASI FOLDER ---
UPLOAD_FOLDER = 'static/uploads'
CAM_FOLDER = 'static/heatmaps'
MODEL_FOLDER = 'models_pytorch' 
TRAIN_DIR = 'dataset/split_80_20/train'
DB_FOLDER = 'static/data/database'  # Folder untuk database CSV

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CAM_FOLDER'] = CAM_FOLDER
app.config['TRAIN_DIR'] = TRAIN_DIR
app.config['DB_FOLDER'] = DB_FOLDER

# Buat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CAM_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get('username'):
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    return wrapper

# --- PROFESSIONAL DATABASE CSV SYSTEM ---
# Menggunakan ISO 8601 timestamp, UUID, dan metadata lengkap

def get_db_path(table_name: str):
    """Mengembalikan path file database CSV berdasarkan nama tabel"""
    return os.path.join(DB_FOLDER, f'{table_name}.csv')

def get_timestamp_iso():
    """Mengembalikan timestamp ISO 8601 dengan timezone UTC"""
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S') + '+0000'

def get_wib_timestamp():
    """Mengembalikan timestamp dalam format WIB (UTC+7)"""
    # Hitung manual: UTC + 7 jam
    utc_time = datetime.utcnow()
    wib_time = utc_time + timedelta(hours=7)
    return wib_time.strftime('%Y-%m-%dT%H:%M:%S+07:00')

# --- TABEL: UPLOADS (Metadata upload) ---
UPLOADS_HEADER = [
    'upload_id', 'timestamp_utc', 'timestamp_wib', 'session_id',
    'original_filename', 'stored_filename', 'file_path',
    'file_size_bytes', 'mime_type', 'image_width', 'image_height',
    'upload_status', 'created_at'
]

def init_uploads_table():
    """Inisialisasi tabel uploads"""
    db_path = get_db_path('uploads')
    if not os.path.exists(db_path):
        with open(db_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(UPLOADS_HEADER)

def log_upload(session_id: str, original_filename: str, stored_filename: str, 
               file_path: str, file_size: int, mime_type: str, 
               width: int, height: int) -> str:
    """Catat upload baru ke database, return upload_id"""
    db_path = get_db_path('uploads')
    if not os.path.exists(db_path):
        init_uploads_table()
    
    upload_id = str(uuid.uuid4())
    timestamp_utc = get_timestamp_iso()
    timestamp_wib = get_wib_timestamp()
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(db_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            upload_id, timestamp_utc, timestamp_wib, session_id,
            original_filename, stored_filename, file_path,
            file_size, mime_type, width, height,
            'pending', created_at
        ])
    
    return upload_id

# --- TABEL: ANALYSES (Hasil analisis) ---
ANALYSES_HEADER = [
    'analysis_id', 'upload_id', 'timestamp_utc', 'timestamp_wib',
    'cloud_coverage_pct', 'oktas_value', 'oktas_description',
    'ensemble_class', 'ensemble_confidence', 'ensemble_method',
    'total_processing_time_ms', 'analysis_status', 'validated_by',
    'validation_notes', 'created_at'
]

def init_analyses_table():
    """Inisialisasi tabel analyses"""
    db_path = get_db_path('analyses')
    if not os.path.exists(db_path):
        with open(db_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(ANALYSES_HEADER)

def log_analysis(upload_id: str, cloud_coverage: float, oktas_val: int,
                 oktas_desc: str, ensemble_class: str, ensemble_conf: float,
                 processing_time: float) -> str:
    """Catat hasil analisis ke database"""
    db_path = get_db_path('analyses')
    if not os.path.exists(db_path):
        init_analyses_table()
    
    analysis_id = str(uuid.uuid4())
    timestamp_utc = get_timestamp_iso()
    timestamp_wib = get_wib_timestamp()
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(db_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            analysis_id, upload_id, timestamp_utc, timestamp_wib,
            round(cloud_coverage, 2), oktas_val, oktas_desc,
            ensemble_class, round(ensemble_conf, 2), 'weighted_voting',
            round(processing_time, 2), 'completed', '', '',
            created_at
        ])
    
    return analysis_id

# --- TABEL: PREDICTIONS (Prediksi per model) ---
PREDICTIONS_HEADER = [
    'prediction_id', 'analysis_id', 'model_name', 'model_version',
    'predicted_class', 'confidence_pct', 'inference_time_ms',
    'class_index', 'top_1_class', 'top_1_conf', 'created_at'
]

def init_predictions_table():
    """Inisialisasi tabel predictions"""
    db_path = get_db_path('predictions')
    if not os.path.exists(db_path):
        with open(db_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(PREDICTIONS_HEADER)

def log_prediction(analysis_id: str, model_name: str, model_version: str,
                   predicted_class: str, confidence: float, inference_time: float,
                   class_idx: int, top1_class: str, top1_conf: float):
    """Catat prediksi individual per model"""
    db_path = get_db_path('predictions')
    if not os.path.exists(db_path):
        init_predictions_table()
    
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(db_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            prediction_id, analysis_id, model_name, model_version,
            predicted_class, round(confidence, 2), round(inference_time, 2),
            class_idx, top1_class, round(top1_conf, 2), created_at
        ])

# --- TABEL: GRADCAM (File heatmap) ---
GRADCAM_HEADER = [
    'gradcam_id', 'analysis_id', 'model_name', 'file_path', 
    'file_url', 'created_at'
]

def init_gradcam_table():
    """Inisialisasi tabel gradcam"""
    db_path = get_db_path('gradcam')
    if not os.path.exists(db_path):
        with open(db_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(GRADCAM_HEADER)

def log_gradcam(analysis_id: str, model_name: str, file_path: str, file_url: str):
    """Catat file gradcam"""
    db_path = get_db_path('gradcam')
    if not os.path.exists(db_path):
        init_gradcam_table()
    
    gradcam_id = str(uuid.uuid4())
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(db_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([gradcam_id, analysis_id, model_name, file_path, file_url, created_at])

# --- TABEL: EVIDENCE (File bukti) ---
EVIDENCE_HEADER = [
    'evidence_id', 'analysis_id', 'source_class', 'file_path',
    'file_url', 'created_at'
]

def init_evidence_table():
    """Inisialisasi tabel evidence"""
    db_path = get_db_path('evidence')
    if not os.path.exists(db_path):
        with open(db_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(EVIDENCE_HEADER)

def log_evidence(analysis_id: str, source_class: str, file_path: str, file_url: str):
    """Catat file evidence"""
    db_path = get_db_path('evidence')
    if not os.path.exists(db_path):
        init_evidence_table()
    
    evidence_id = str(uuid.uuid4())
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(db_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([evidence_id, analysis_id, source_class, file_path, file_url, created_at])

# --- MODEL VERSIONS ---
MODEL_VERSIONS = {
    'Simple CNN': 'v2.0',
    'MobileNetV2': 'finetuned_v1',
    'EfficientNetB0': 'finetuned_v1'
}

# Inisialisasi semua tabel saat startup
init_uploads_table()
init_analyses_table()
init_predictions_table()
init_gradcam_table()
init_evidence_table()

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
# --- GANTI WILAYAH_DICT DI app.py ---
WILAYAH_DICT = {
    # JAKARTA
    "31.71.03.1001": "STAMET KEMAYORAN (JAKARTA PUSAT)",
    "31.72.03.1001": "STAMET TANJUNG PRIOK (JAKARTA UTARA)",
    "31.75.06.1002": "STAMET HALIM PK (JAKARTA TIMUR)",
    
    # TANGERANG & BANTEN
    "36.74.03.1001": "STAKLIM BANTEN (PONDOK BETUNG)",
    "36.71.02.1002": "STAMET SOEKARNO-HATTA (CENGKARENG)",
    "36.03.12.2001": "STAMET BUDIARTO (CURUG/TANGERANG)",
    
    # BOGOR & BEKASI
    "32.01.23.2003": "STAMET CITEKO (CISARUA/PUNCAK)",
    "32.71.04.1001": "STAKLIM JAWA BARAT (DRAMAGA/BOGOR)",
    "32.16.20.2005": "POS PENGAMATAN CIKARANG (BEKASI)"
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
        path = os.path.join(MODEL_FOLDER, 'simple_cnn_v2_80_20.pth')
        if not os.path.exists(path):
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
        path = os.path.join(MODEL_FOLDER, 'mobilenet_finetuned_80_20.pth')
        if not os.path.exists(path):
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
        path = os.path.join(MODEL_FOLDER, 'efficientnet_finetuned_80_20.pth')
        if not os.path.exists(path):
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

# --- ROUTE LOGIN ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('username'):
        return redirect(url_for('index'))

    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        user = USERS.get(username)

        if user and user['password'] == password:
            session['username'] = username
            session['role'] = user['role']
            return redirect(url_for('index'))
        else:
            error = 'Username atau password tidak valid. Gunakan admin/admin atau user/user.'

    return render_template('login.html', error=error)


# --- BILINGUAL SUPPORT: Load language JSON files ---
def load_language(lang_code: str):
    """Load language JSON from static/data/lang_{code}.json"""
    if lang_code not in ('en', 'id'):
        lang_code = 'en'
    path = os.path.join('static', 'data', f'lang_{lang_code}.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def perform_analysis(filepath, filename=None):
    """Run cloud analysis + model inference and return structured results."""
    results = []
    heatmap_data = {}
    final_decision = None
    impact_info = None
    evidence_files = []
    cloud_coverage = 0
    oktas_val = 0
    total_processing_time = 0

    try:
        cloud_coverage, oktas_val = analyze_cloud_properties(filepath)

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

                latency = round((time.time() - start_time) * 1000, 2)
                results.append({'model': name, 'prediction': prediction, 'confidence': round(score * 100, 2), 'latency': latency})
                total_processing_time += latency

                if prediction not in vote_box: vote_box[prediction] = []
                vote_box[prediction].append(score)

                # Grad-CAM (best-effort)
                target_layer = None
                if name == 'Simple CNN': target_layer = model.features[-2]
                elif name == 'MobileNetV2': target_layer = model.features[-1]
                elif name == 'EfficientNetB0': target_layer = model.features[-1]

                if target_layer and filename:
                    cam_extractor = GradCAM(model, target_layer)
                    heatmap = cam_extractor(img_tensor, class_idx=idx)
                    cam_path = os.path.join(app.config['CAM_FOLDER'], f"cam_{name}_{filename}")
                    save_gradcam(filepath, heatmap, cam_path)
                    heatmap_data[name] = {'img': url_for('static', filename=f'heatmaps/cam_{name}_{filename}'), 'conf': round(score*100, 2), 'pred': prediction}
            except Exception as e:
                print(f"[ERR] Inference {name}: {e}")

        # Ensemble Weighted Voting
        best_class = None
        highest_score = -1
        for cls, scores in vote_box.items():
            weighted = (sum(scores)/len(scores)) + (len(scores) * 0.15)
            if weighted > highest_score:
                highest_score = weighted
                best_class = cls

        if best_class:
            final_decision = {'class': best_class, 'confidence': round((sum(vote_box[best_class])/len(vote_box[best_class])) * 100, 2)}
            if REPORT_FEATURE_ACTIVE:
                try:
                    impact_info = get_weather_impact(best_class)
                except:
                    impact_info = None
            # evidence sampling (best-effort)
            try:
                class_path = os.path.join(app.config['TRAIN_DIR'], best_class)
                if os.path.exists(class_path):
                    all_imgs = [i for i in os.listdir(class_path) if i.lower().endswith(('.png','.jpg'))]
                    evidence_files = [f"{best_class}/{i}" for i in random.sample(all_imgs, min(len(all_imgs), 3))]
            except:
                evidence_files = []

    except Exception as e:
        print(f"[ERR] perform_analysis: {e}")

    return {
        'results': results,
        'final_decision': final_decision,
        'heatmap_data': heatmap_data,
        'impact_info': impact_info,
        'evidence_files': evidence_files,
        'cloud_coverage': cloud_coverage,
        'oktas_val': oktas_val,
        'processing_time': total_processing_time
    }



@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/neural_sky', methods=['GET', 'POST'])
@login_required
def neural_sky():
    lang = request.args.get('lang', 'en')
    texts = load_language(lang)
    report_file = None
    analysis = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('neural_sky.html', texts=texts, error=texts.get('no_file', 'No file'))
        file = request.files['file']
        if file.filename == '':
            return render_template('neural_sky.html', texts=texts, error=texts.get('no_file', 'No file'))

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = file.filename
        ext = os.path.splitext(original_filename)[1].lower()
        if ext == '': ext = '.jpg'
        filename = f"cloud_{timestamp}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Log upload minimal metadata
        try:
            session_id = request.headers.get('X-Session-ID', str(uuid.uuid4()))
            log_upload(session_id, original_filename, filename, filepath, os.path.getsize(filepath), 'image', *Image.open(filepath).size)
        except:
            pass

        analysis = perform_analysis(filepath, filename)

    return render_template('neural_sky.html', texts=texts, analysis=analysis, lang=lang)


@app.route('/neural_sky/report', methods=['POST'])
@login_required
def neural_sky_report():
    lang = request.form.get('lang', 'en')
    texts = load_language(lang)
    # Expect simple fields posted from client
    filename = request.form.get('filename')
    classname = request.form.get('class')
    confidence = request.form.get('confidence')
    coverage = request.form.get('coverage')
    oktas = request.form.get('oktas')

    # Build a simple translated text report
    lines = []
    lines.append(texts.get('report_title', 'Neural Sky Report'))
    lines.append('---')
    lines.append(f"{texts.get('file', 'File')}: {filename}")
    lines.append(f"{texts.get('predicted_class', 'Predicted class')}: {classname}")
    lines.append(f"{texts.get('confidence', 'Confidence')}: {confidence}")
    lines.append(f"{texts.get('cloud_coverage', 'Cloud coverage')}: {coverage}")
    lines.append(f"{texts.get('oktas', 'Oktas')}: {oktas}")
    report_text = '\n'.join(lines)

    # Save to temp file and send
    tmp_path = os.path.join('static', 'data', f"report_{int(time.time())}.txt")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    return send_file(tmp_path, as_attachment=True, download_name=f"NeuralSky_Report_{filename or 'result'}.txt")


# --- ROUTE UTAMA (DASHBOARD) ---
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    # language preference from session or query
    lang = request.args.get('lang') or session.get('lang', 'id')
    session['lang'] = lang
    texts = load_language(lang)
    # Inisialisasi variabel untuk tracking
    current_upload_id = None
    current_analysis_id = None
    
    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        if file.filename == '': return redirect(request.url)
        if file:
            # Simpan timestamp untuk penamaan file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = file.filename
            
            # Ekstrak ekstensi file
            ext = os.path.splitext(original_filename)[1].lower()
            if ext == '': ext = '.jpg'
            
            # Buat filename dengan timestamp: cloud_YYYYMMDD_HHMMSS_ext
            filename = f"cloud_{timestamp}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Ambil metadata file
            file_size = os.path.getsize(filepath)
            mime_type = f"image/{ext[1:]}" if ext[1:] in ['jpg', 'jpeg', 'png', 'gif'] else 'application/octet-stream'
            
            # Ambil dimensi gambar
            try:
                with Image.open(filepath) as img:
                    img_width, img_height = img.size
            except:
                img_width, img_height = 0, 0
            
            # Generate session ID (atau gunakan yang ada)
            session_id = request.headers.get('X-Session-ID', str(uuid.uuid4()))
            
            # Catat ke database uploads (tabel profesional)
            current_upload_id = log_upload(
                session_id=session_id,
                original_filename=original_filename,
                stored_filename=filename,
                file_path=filepath,
                file_size=file_size,
                mime_type=mime_type,
                width=img_width,
                height=img_height
            )
            
            print(f"[INFO] Upload recorded: {current_upload_id} - {filename}")
            
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
            
            # Hitung total processing time
            total_processing_time = sum(r['latency'] for r in results)
            
            # Deskripsi oktas
            oktas_descriptions = {
                0: 'Clear Sky', 1: 'Trace', 2: 'Few', 3: 'Scattered',
                4: 'Broken', 5: 'Overcast', 6: 'Obscured', 7: 'Overcast (Partial)',
                8: 'Overcast (Complete)'
            }
            oktas_desc = oktas_descriptions.get(oktas_val, 'Unknown')
            
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
                
                # === PROFESSIONAL DATABASE LOGGING ===
                # Cari upload_id dari filename
                upload_id = None
                uploads_path = get_db_path('uploads')
                if os.path.exists(uploads_path):
                    with open(uploads_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row['stored_filename'] == filename:
                                upload_id = row['upload_id']
                                break
                
                if upload_id:
                    # Log ke tabel analyses
                    current_analysis_id = log_analysis(
                        upload_id=upload_id,
                        cloud_coverage=cloud_coverage,
                        oktas_val=oktas_val,
                        oktas_desc=oktas_desc,
                        ensemble_class=best_class,
                        ensemble_conf=final_decision['confidence'],
                        processing_time=total_processing_time
                    )
                    
                    # Log prediksi per model
                    for r in results:
                        model_ver = MODEL_VERSIONS.get(r['model'], 'unknown')
                        log_prediction(
                            analysis_id=current_analysis_id,
                            model_name=r['model'],
                            model_version=model_ver,
                            predicted_class=r['prediction'],
                            confidence=r['confidence'],
                            inference_time=r['latency'],
                            class_idx=CLASS_NAMES.index(r['prediction']) if r['prediction'] in CLASS_NAMES else 0,
                            top1_class=r['prediction'],
                            top1_conf=r['confidence']
                        )
                    
                    # Log gradcam files
                    for name, data in heatmap_data.items():
                        gradcam_path = os.path.join(app.config['CAM_FOLDER'], f"cam_{name}_{filename}")
                        gradcam_url = url_for('static', filename=f'heatmaps/cam_{name}_{filename}')
                        log_gradcam(
                            analysis_id=current_analysis_id,
                            model_name=name,
                            file_path=gradcam_path,
                            file_url=gradcam_url
                        )
                    
                    # Log evidence files
                    for ev_file in evidence_files:
                        ev_path = os.path.join(app.config['TRAIN_DIR'], ev_file)
                        ev_url = url_for('dataset_image', filename=ev_file)
                        log_evidence(
                            analysis_id=current_analysis_id,
                            source_class=best_class,
                            file_path=ev_path,
                            file_url=ev_url
                        )
                    
                    print(f"[INFO] Analysis recorded: {current_analysis_id} for upload {upload_id}")

    return render_template('index.html',
                           results=results, image_file=image_url,
                           final_decision=final_decision, impact_info=impact_info,
                           heatmap_data=heatmap_data, evidence_files=evidence_files,
                           class_names=CLASS_NAMES, report_active=REPORT_FEATURE_ACTIVE,
                           cloud_coverage=cloud_coverage, oktas_val=oktas_val,
                           user=session.get('username'), role=session.get('role'),
                           texts=texts, lang=lang)

# --- ROUTE UTILS ---
@app.route('/dataset_image/<path:filename>')
@login_required
def dataset_image(filename):
    return send_file(os.path.join(app.config['TRAIN_DIR'], filename))

@app.route('/generate_report', methods=['POST'])
@login_required
def generate_report():
    if not REPORT_FEATURE_ACTIVE: return "Feature Disabled", 500
    filename = request.form.get('filename')
    ai_recommendation = request.form.get('ai_recommendation')
    corrected_class = request.form.get('corrected_class')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not corrected_class or corrected_class.strip() == '':
        corrected_class = ai_recommendation or 'clear sky'

    # Jika pengguna tidak melakukan override manual, laporan mengikuti rekomendasi AI.
    if corrected_class == ai_recommendation:
        confidence_str = "AI RECOMMENDATION"
    else:
        confidence_str = "MANUAL VALIDATION"

    coverage, oktas = analyze_cloud_properties(filepath)
    impacts = get_weather_impact(corrected_class)

    heatmap_path = f"static/heatmaps/cam_EfficientNetB0_{filename}"
    if not os.path.exists(heatmap_path): heatmap_path = filepath

    pdf_path = create_pdf(filename, corrected_class, confidence_str, impacts, heatmap_path, coverage, oktas)
    return send_file(pdf_path, as_attachment=True, download_name=f"Laporan_{corrected_class}_{filename.split('.')[0]}.pdf")


@app.route('/set_lang')
def set_lang():
    lang = request.args.get('lang')
    if lang in ('en', 'id'):
        session['lang'] = lang
    # redirect back to referrer or index
    ref = request.headers.get('Referer') or url_for('index')
    return redirect(ref)

# --- ROUTE DATA BMKG (UPDATED WITH DROPDOWN LOGIC) ---
@app.route('/bmkg_feed')
@login_required
def bmkg_feed():
    try:
        # GANTI DEFAULT DARI PAMULANG KE KEMAYORAN (31.71.03.1001)
        kode = request.args.get('kode', "31.71.03.1001") 
        
        # Cari Nama Kota
        nama_kota = WILAYAH_DICT.get(kode, "Lokasi Terpilih")

        # Ambil Data
        weather_data = get_bmkg_data(kode)
        
        # URL Satelit Default (Indonesia)
        import time
        ts = int(time.time())
        satellite_url = f"https://inderaja.bmkg.go.id/IMAGE/HIMA/H08_EH_Indonesia.png?v={ts}"
        
        lokasi_info = {}
        flat_forecast = []
        
        if weather_data and 'lokasi' in weather_data:
            lokasi_info = weather_data['lokasi']
            if 'data' in weather_data and weather_data['data']:
                cuaca_per_hari = weather_data['data'][0]['cuaca']
                for hari in cuaca_per_hari:
                    for jam in hari:
                        flat_forecast.append(jam)
        
        return render_template('components/tab_bmkg.html', 
                               lokasi=lokasi_info, 
                               forecasts=flat_forecast,
                               satellite=satellite_url,
                               daftar_wilayah=WILAYAH_DICT,
                               current_kode=kode,
                               current_nama=nama_kota)
    except Exception as e:
        print(f"[ERR] BMKG Feed: {e}")
        return f"<div class='text-danger p-5'>Error: {e}</div>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)