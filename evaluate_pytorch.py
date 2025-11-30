import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# --- KONFIGURASI ---
TEST_DIR = 'dataset/clouds_test'
MODEL_PATH = 'models_pytorch'
STATIC_PATH = 'static'
IMG_SIZE = 224
BATCH_SIZE = 32

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Evaluasi berjalan di: {device}")

# --- ARSITEKTUR MODEL (Harus sama persis dengan train_pytorch.py) ---
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
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- HELPER UTAMA ---
def get_test_loader():
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader, dataset.classes

def evaluate_and_plot(model, model_name, save_filename, test_loader, class_names):
    print(f"\n--- Menguji {model_name} ---")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 1. Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # 2. Plotting (Gaya Dark Mode / Sci-Fi)
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    
    # Colormap 'mako' atau 'crest' sangat cocok untuk tema Neon Cyan
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako', 
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, square=True,
                annot_kws={"size": 14, "weight": "bold", "color": "white"},
                linewidths=1, linecolor='#050810') # Warna border gelap
    
    plt.title(f'{model_name.upper()}', fontsize=20, color='#00f2ff', pad=20, weight='bold') # Judul Neon
    plt.ylabel('TRUE LABEL', fontsize=12, color='gray')
    plt.xlabel('PREDICTED LABEL', fontsize=12, color='gray')
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(rotation=0, color='white')
    
    plt.tight_layout()
    
    # 3. Simpan Gambar
    save_path = os.path.join(STATIC_PATH, save_filename)
    plt.savefig(save_path, transparent=True, dpi=150)
    plt.close()
    print(f"[SUKSES] Grafik tersimpan: {save_path}")
    
    # 4. Print Report Singkat di Console
    print(classification_report(all_labels, all_preds, target_names=class_names))

# --- EKSEKUSI ---
if __name__ == "__main__":
    loader, class_names = get_test_loader()
    num_classes = len(class_names)
    
    # 1. Evaluate Simple CNN
    try:
        model = SimpleCNN(num_classes)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'simple_cnn_v2.pth')))
        model.to(device)
        evaluate_and_plot(model, "Simple CNN", "confusion_simple_cnn_v2.png", loader, class_names)
    except Exception as e: print(f"Skip Simple CNN: {e}")

    # 2. Evaluate MobileNetV2
    try:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'mobilenet_finetuned.pth')))
        model.to(device)
        evaluate_and_plot(model, "MobileNet V2", "confusion_mobilenet_finetuned.png", loader, class_names)
    except Exception as e: print(f"Skip MobileNet: {e}")

    # 3. Evaluate EfficientNetB0
    try:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(1280, num_classes)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'efficientnet_finetuned.pth')))
        model.to(device)
        evaluate_and_plot(model, "EfficientNet B0", "confusion_efficientnet_finetuned.png", loader, class_names)
    except Exception as e: print(f"Skip EfficientNet: {e}")