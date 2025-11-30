import os
import time
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, url_for, redirect, send_file
import numpy as np
import cv2  # <--- LIBRARY BARU

# --- IMPORT MODUL CUSTOM ---
from gradcam_pytorch import GradCAM, save_gradcam 

try:
    from impact_logic import get_weather_impact
    from report_generator import create_pdf
    REPORT_FEATURE_ACTIVE = True
except ImportError:
    REPORT_FEATURE_ACTIVE = False
    print("[WARN] Modul Impact/Report tidak ditemukan.")

app = Flask(__name__)

# --- KONFIGURASI ---
UPLOAD_FOLDER = 'static/uploads'
CAM_FOLDER = 'static/heatmaps'
MODEL_FOLDER = 'models_pytorch' 
TRAIN_DIR = 'dataset/clouds_train'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CAM_FOLDER'] = CAM_FOLDER
app.config['TRAIN_DIR'] = TRAIN_DIR

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CAM_FOLDER, exist_ok=True)

# --- SETUP DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Server berjalan di: {device}")

# --- DETEKSI KELAS ---
if os.path.exists(TRAIN_DIR):
    CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))
else:
    CLASS_NAMES = ['cirriform clouds', 'clear sky', 'cumulonimbus clouds', 
                   'cumulus clouds', 'high cumuliform clouds', 
                   'stratiform clouds', 'stratocumulus clouds']

NUM_CLASSES = len(CLASS_NAMES)

# --- DEFINISI MODEL ---
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
    try:
        model = SimpleCNN(NUM_CLASSES)
        path = os.path.join(MODEL_FOLDER, 'simple_cnn_v2.pth')
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device).eval()
            models_dict['Simple CNN'] = model
    except Exception: pass

    try:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
        path = os.path.join(MODEL_FOLDER, 'mobilenet_finetuned.pth')
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device).eval()
            models_dict['MobileNetV2'] = model
    except Exception: pass

    try:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
        path = os.path.join(MODEL_FOLDER, 'efficientnet_finetuned.pth')
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device).eval()
            models_dict['EfficientNetB0'] = model
    except Exception: pass
load_models()

# --- HELPER ---
val_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- FUNGSI BARU: HITUNG OKTAS (COMPUTER VISION) ---
def analyze_cloud_properties(filepath):
    try:
        img = cv2.imread(filepath)
        if img is None: return 0, 0
        
        # Grayscale & Thresholding (Memisahkan awan putih dari langit biru/gelap)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Threshold 110: Angka eksperimental (awan biasanya pixel > 110)
        _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        
        total_pixels = img.shape[0] * img.shape[1]
        cloud_pixels = cv2.countNonZero(thresh)
        
        coverage_percent = (cloud_pixels / total_pixels) * 100
        
        # Konversi ke Oktas (0-8)
        oktas = round((coverage_percent / 100) * 8)
        
        return round(coverage_percent, 1), oktas
    except Exception as e:
        print(f"[ERR] Gagal hitung oktas: {e}")
        return 0, 0

# --- ROUTES ---
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
    results = []
    final_decision = None
    heatmap_data = {}
    impact_info = None
    image_url = None
    evidence_files = []
    
    # Variabel Oktas
    cloud_coverage = 0
    oktas_val = 0

    if filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            image_url = url_for('static', filename=f'uploads/{filename}')
            
            # 1. Hitung Oktas (Fitur Baru)
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

                    # Grad-CAM Logic
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
                except Exception: pass

            # Ensemble
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
                    impact_info = get_weather_impact(best_class)
                    # Evidence Retrieval
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
                           # Kirim data oktas ke HTML
                           cloud_coverage=cloud_coverage, oktas_val=oktas_val)

@app.route('/dataset_image/<path:filename>')
def dataset_image(filename):
    return send_file(os.path.join(app.config['TRAIN_DIR'], filename))

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if not REPORT_FEATURE_ACTIVE: return "Feature Disabled", 500
    filename = request.form.get('filename')
    corrected_class = request.form.get('corrected_class')
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    model = models_dict.get('EfficientNetB0')
    confidence_str = ""
    
    # Hitung ulang oktas untuk laporan
    coverage, oktas = analyze_cloud_properties(filepath)

    if model:
        img_pil = Image.open(filepath).convert('RGB')
        img_tensor = val_transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_tensor)
            ai_score = torch.max(torch.nn.functional.softmax(out, dim=1), 1)[0].item()
            ai_pred = CLASS_NAMES[torch.argmax(out).item()]
            
            if corrected_class == ai_pred:
                confidence_str = f"{round(ai_score * 100, 2)}% (AI Confidence)"
            else:
                confidence_str = "MANUAL VALIDATION (Expert Override)"

    impacts = get_weather_impact(corrected_class)
    heatmap_path = f"static/heatmaps/cam_EfficientNetB0_{filename}"
    if not os.path.exists(heatmap_path): heatmap_path = filepath
    
    # Update fungsi create_pdf dengan parameter oktas
    pdf_path = create_pdf(filename, corrected_class, confidence_str, impacts, heatmap_path, coverage, oktas)
    return send_file(pdf_path, as_attachment=True, download_name=f"Laporan_{corrected_class}_{filename.split('.')[0]}.pdf")

if __name__ == '__main__':
    app.run(debug=True, port=5000)