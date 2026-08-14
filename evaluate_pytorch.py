import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import os

# --- KONFIGURASI ---
TEST_DIR = 'dataset/split_80_20/test'
MODEL_PATH = 'models_pytorch'
STATIC_PATH = 'static'
SUMMARY_CSV = os.path.join(STATIC_PATH, 'data', 'metrics_summary.csv')
IMG_SIZE = 224
BATCH_SIZE = 32
ENSEMBLE_WEIGHTS = {
    'Simple CNN': 0.3,
    'MobileNet V2': 0.3,
    'EfficientNet B0': 0.4
}

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
    # 2. Compute numeric metrics: per-class and overall
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names))), zero_division=0
    )
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))

    # Prepare paths
    os.makedirs(os.path.join(STATIC_PATH, 'data'), exist_ok=True)
    metrics_csv = os.path.join(STATIC_PATH, 'data', f'metrics_{model_name.lower().replace(" ", "_")}.csv')
    summary_csv = SUMMARY_CSV

    # Save per-class metrics
    with open(metrics_csv, 'w', encoding='utf-8') as f:
        f.write('class,precision,recall,f1,support\n')
        for i, name in enumerate(class_names):
            f.write(f'{name},{precision[i]:.4f},{recall[i]:.4f},{f1[i]:.4f},{int(support[i])}\n')
        # macro averages
        macro_p = np.mean(precision)
        macro_r = np.mean(recall)
        macro_f = np.mean(f1)
        f.write(f'MACRO_AVG,{macro_p:.4f},{macro_r:.4f},{macro_f:.4f},{int(np.sum(support))}\n')
    print(f"[SUKSES] Metrics CSV tersimpan: {metrics_csv}")

    # Append model-level summary
    header = 'model,accuracy,precision_macro,recall_macro,f1_macro\n'
    line = f'{model_name},{accuracy:.4f},{macro_p:.4f},{macro_r:.4f},{macro_f:.4f}\n'
    if os.path.exists(summary_csv):
        existing = {}
        with open(summary_csv, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if lines:
            for row in lines[1:]:
                parts = row.split(',')
                if len(parts) >= 5:
                    existing[parts[0]] = row
        existing[model_name] = line.strip()
        with open(summary_csv, 'w', encoding='utf-8') as f:
            f.write(header)
            for model_key in existing:
                f.write(existing[model_key] + '\n')
    else:
        with open(summary_csv, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(line)

    # 3. Plotting: create both dark and light confusion matrix images
    # Dark version (existing style)
    try:
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='mako',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=False, square=True,
                    annot_kws={"size": 14, "weight": "bold", "color": "white"},
                    linewidths=1, linecolor='#050810')
        plt.title(f'{model_name.upper()}', fontsize=20, color='#00f2ff', pad=20, weight='bold')
        plt.ylabel('True Label (Kelas)', fontsize=12, color='white')
        plt.xlabel('Predicted Label (Kelas)', fontsize=12, color='white')
        plt.xticks(rotation=45, ha='right', color='white')
        plt.yticks(rotation=0, color='white')
        plt.tight_layout()
        dark_path = os.path.join(STATIC_PATH, save_filename)
        plt.savefig(dark_path, transparent=True, dpi=150)
        plt.close()
        print(f"[SUKSES] Grafik dark tersimpan: {dark_path}")
    except Exception as e:
        print(f"[WARN] Gagal menyimpan dark CM: {e}")

    # Light version (white background)
    try:
        style_name = 'seaborn' if 'seaborn' in plt.style.available else 'default'
        plt.style.use(style_name)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=False, square=True,
                    annot_kws={"size": 12, "weight": "bold", "color": "black"},
                    linewidths=0.8, linecolor='gray', ax=ax)
        ax.set_title(f'{model_name.upper()}', fontsize=18, weight='bold')
        ax.set_ylabel('True Label (Kelas)')
        ax.set_xlabel('Predicted Label (Kelas)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        light_path = os.path.join(STATIC_PATH, save_filename.replace('.png', '_light.png'))
        fig.savefig(light_path, dpi=150, facecolor='white')
        plt.close(fig)
        print(f"[SUKSES] Grafik light tersimpan: {light_path}")
    except Exception as e:
        print(f"[WARN] Gagal menyimpan light CM: {e}")
    
    # 4. Print Report Singkat di Console
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 5. Save per-class precision/recall/f1 visualization
    plot_per_class_metrics(class_names, all_labels, all_preds, model_name)

    # 6. Save per-class metrics table visualization
    plot_metrics_table(model_name, class_names, precision, recall, f1, support, accuracy, macro_p, macro_r, macro_f)


def plot_metrics_table(model_name, class_names, precision, recall, f1, support, accuracy, macro_p, macro_r, macro_f):
    rows = []
    for i, name in enumerate(class_names):
        rows.append([name, f'{precision[i]:.2f}', f'{recall[i]:.2f}', f'{f1[i]:.2f}', str(int(support[i]))])
    rows.append(['MACRO AVG', f'{macro_p:.2f}', f'{macro_r:.2f}', f'{macro_f:.2f}', str(int(np.sum(support)))])
    rows.append(['ACCURACY', f'{accuracy:.2f}', '', '', ''])

    col_labels = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    fig_height = max(4, 0.4 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height), facecolor='white')
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4f81bd')
        else:
            cell.set_facecolor('#f8f8f8' if row % 2 == 0 else 'white')

    ax.set_title(f'{model_name} - Metrics Table', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    save_path = os.path.join(STATIC_PATH, 'data', f'metrics_table_{model_name.lower().replace(" ", "_")}.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SUKSES] Metrics table visualization tersimpan: {save_path}")


def plot_summary_table(summary_csv):
    if not os.path.exists(summary_csv):
        return

    with open(summary_csv, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if len(lines) < 2:
        return

    headers = lines[0].split(',')
    rows = [line.split(',') for line in lines[1:]]
    fig_height = max(2.5, 0.8 * len(rows) + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height), facecolor='white')
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4f81bd')
        else:
            cell.set_facecolor('#f8f8f8' if row % 2 == 0 else 'white')

    ax.set_title('Model Summary Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    save_path = os.path.join(STATIC_PATH, 'data', 'metrics_summary_table.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SUKSES] Summary metrics table tersimpan: {save_path}")


def get_ensemble_predictions(models, weights, test_loader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            ensemble_probs = None
            for name, model in models.items():
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1) * weights[name]
                ensemble_probs = probs if ensemble_probs is None else ensemble_probs + probs

            _, preds = torch.max(ensemble_probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds


def evaluate_ensemble(models, weights, loader, class_names):
    print("\n--- Menguji Hybrid Ensemble ---")
    all_labels, all_preds = get_ensemble_predictions(models, weights, loader)
    cm = confusion_matrix(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names))), zero_division=0
    )
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))

    os.makedirs(os.path.join(STATIC_PATH, 'data'), exist_ok=True)
    metrics_csv = os.path.join(STATIC_PATH, 'data', 'metrics_hybrid_ensemble.csv')

    with open(metrics_csv, 'w', encoding='utf-8') as f:
        f.write('class,precision,recall,f1,support\n')
        for i, name in enumerate(class_names):
            f.write(f'{name},{precision[i]:.4f},{recall[i]:.4f},{f1[i]:.4f},{int(support[i])}\n')
        macro_p = np.mean(precision)
        macro_r = np.mean(recall)
        macro_f = np.mean(f1)
        f.write(f'MACRO_AVG,{macro_p:.4f},{macro_r:.4f},{macro_f:.4f},{int(np.sum(support))}\n')
    print(f"[SUKSES] Metrics CSV tersimpan: {metrics_csv}")

    summary_csv = SUMMARY_CSV
    header = 'model,accuracy,precision_macro,recall_macro,f1_macro\n'
    line = f'Hybrid Ensemble,{accuracy:.4f},{macro_p:.4f},{macro_r:.4f},{macro_f:.4f}\n'
    if os.path.exists(summary_csv):
        existing = {}
        with open(summary_csv, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if lines:
            for row in lines[1:]:
                parts = row.split(',')
                if len(parts) >= 5:
                    existing[parts[0]] = row
        existing['Hybrid Ensemble'] = line.strip()
        with open(summary_csv, 'w', encoding='utf-8') as f:
            f.write(header)
            for model_key in existing:
                f.write(existing[model_key] + '\n')
    else:
        with open(summary_csv, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(line)

    # Create confusion matrix plots for ensemble
    try:
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='mako',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=False, square=True,
                    annot_kws={"size": 14, "weight": "bold", "color": "white"},
                    linewidths=1, linecolor='#050810')
        plt.title('HYBRID ENSEMBLE', fontsize=20, color='#00f2ff', pad=20, weight='bold')
        plt.ylabel('True Label (Kelas)', fontsize=12, color='white')
        plt.xlabel('Predicted Label (Kelas)', fontsize=12, color='white')
        plt.xticks(rotation=45, ha='right', color='white')
        plt.yticks(rotation=0, color='white')
        plt.tight_layout()
        dark_path = os.path.join(STATIC_PATH, 'confusion_hybrid_ensemble.png')
        plt.savefig(dark_path, transparent=True, dpi=150)
        plt.close()
        print(f"[SUKSES] Grafik dark tersimpan: {dark_path}")
    except Exception as e:
        print(f"[WARN] Gagal menyimpan dark CM ensemble: {e}")

    try:
        style_name = 'seaborn' if 'seaborn' in plt.style.available else 'default'
        plt.style.use(style_name)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=False, square=True,
                    annot_kws={"size": 12, "weight": "bold", "color": "black"},
                    linewidths=0.8, linecolor='gray', ax=ax)
        ax.set_title('HYBRID ENSEMBLE', fontsize=18, weight='bold')
        ax.set_ylabel('True Label (Kelas)')
        ax.set_xlabel('Predicted Label (Kelas)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        light_path = os.path.join(STATIC_PATH, 'confusion_hybrid_ensemble_light.png')
        fig.savefig(light_path, dpi=150, facecolor='white')
        plt.close(fig)
        print(f"[SUKSES] Grafik light tersimpan: {light_path}")
    except Exception as e:
        print(f"[WARN] Gagal menyimpan light CM ensemble: {e}")

    print(classification_report(all_labels, all_preds, target_names=class_names))
    plot_metrics_table('Hybrid Ensemble', class_names, precision, recall, f1, support, accuracy, macro_p, macro_r, macro_f)


def plot_per_class_metrics(class_names, y_true, y_pred, model_name):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    x = np.arange(len(class_names))
    width = 0.25
    # Choose a plotting style from a prioritized list, fall back safely
    preferred_styles = ['seaborn-whitegrid', 'seaborn', 'ggplot', 'default']
    for s in preferred_styles:
        if s in plt.style.available:
            plt.style.use(s)
            break
    else:
        plt.style.use('default')
    plt.figure(figsize=(14, 6), facecolor='white')

    # High-contrast, colorblind-friendly palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Draw bars with strong edges for contrast
    bars_precision = plt.bar(x - width, precision, width, label='Precision', color=colors[0], edgecolor='black', linewidth=0.7)
    bars_recall = plt.bar(x, recall, width, label='Recall', color=colors[1], edgecolor='black', linewidth=0.7)
    bars_f1 = plt.bar(x + width, f1, width, label='F1-Score', color=colors[2], edgecolor='black', linewidth=0.7)

    plt.xlabel('Kelas Awan', fontsize=14, fontweight='semibold', color='black')
    plt.ylabel('Skor (Precision / Recall / F1)', fontsize=14, fontweight='semibold', color='black')
    plt.title(f'Per-Class Performance Metrics - {model_name}', fontsize=18, fontweight='bold', color='black')
    plt.xticks(x, [name.replace('_', ' ').title() for name in class_names], rotation=30, ha='right', fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5, color='#888888')
    leg = plt.legend(loc='upper right', frameon=True, fontsize=12)

    # Helper: decide annotation color based on bar color luminance
    def _annotation_color(hex_color):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16)/255.0, int(h[2:4], 16)/255.0, int(h[4:6], 16)/255.0
        luminance = 0.299*r + 0.587*g + 0.114*b
        return 'white' if luminance < 0.5 else 'black'

    # Annotate bars with scores (formatted as 0.00)
    for bars, col in zip([bars_precision, bars_recall, bars_f1], colors):
        ann_color = _annotation_color(col)
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 6),
                         textcoords='offset points',
                         ha='center', va='bottom', fontsize=10, color=ann_color, fontweight='semibold')

    plt.tight_layout()

    save_dir = 'static/training_history'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'per_class_metrics_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SUKSES] Per-class metrics visualization tersimpan: {save_path}")

# --- EKSEKUSI ---
if __name__ == "__main__":
    loader, class_names = get_test_loader()
    num_classes = len(class_names)
    
    # 1. Evaluate Simple CNN
    try:
        model = SimpleCNN(num_classes)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'simple_cnn_v2_80_20.pth')))
        model.to(device)
        evaluate_and_plot(model, "Simple CNN", "confusion_simple_cnn_v2_80_20.png", loader, class_names)
    except Exception as e: print(f"Skip Simple CNN: {e}")

    # 2. Evaluate MobileNetV2
    try:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'mobilenet_finetuned_80_20.pth')))
        model.to(device)
        evaluate_and_plot(model, "MobileNet V2", "confusion_mobilenet_finetuned_80_20.png", loader, class_names)
    except Exception as e: print(f"Skip MobileNet: {e}")

    loaded_models = {}

    # 1. Evaluate Simple CNN
    try:
        model = SimpleCNN(num_classes)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'simple_cnn_v2_80_20.pth')))
        model.to(device)
        evaluate_and_plot(model, "Simple CNN", "confusion_simple_cnn_v2_80_20.png", loader, class_names)
        loaded_models['Simple CNN'] = model
    except Exception as e: print(f"Skip Simple CNN: {e}")

    # 2. Evaluate MobileNetV2
    try:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'mobilenet_finetuned_80_20.pth')))
        model.to(device)
        evaluate_and_plot(model, "MobileNet V2", "confusion_mobilenet_finetuned_80_20.png", loader, class_names)
        loaded_models['MobileNet V2'] = model
    except Exception as e: print(f"Skip MobileNet: {e}")

    # 3. Evaluate EfficientNetB0
    try:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(1280, num_classes)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'efficientnet_finetuned_80_20.pth')))
        model.to(device)
        evaluate_and_plot(model, "EfficientNet B0", "confusion_efficientnet_finetuned_80_20.png", loader, class_names)
        loaded_models['EfficientNet B0'] = model
    except Exception as e: print(f"Skip EfficientNet: {e}")

    # 4. Evaluate Hybrid Ensemble if all base models are available
    if len(loaded_models) == 3:
        evaluate_ensemble(loaded_models, ENSEMBLE_WEIGHTS, loader, class_names)
    else:
        print('[WARN] Hybrid Ensemble evaluation skipped: tidak semua model berhasil dimuat.')

    # 5. Create summary table image for all evaluated models
    plot_summary_table(SUMMARY_CSV)