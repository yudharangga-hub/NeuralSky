import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import time
import os
import copy
import matplotlib.pyplot as plt # <--- TAMBAHAN PENTING

# --- KONFIGURASI ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
TRAIN_DIR = 'dataset/clouds_train'
TEST_DIR = 'dataset/clouds_test'
MODEL_SAVE_PATH = 'models_pytorch'
HISTORY_SAVE_PATH = 'static/training_history' # <--- Folder simpan grafik

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(HISTORY_SAVE_PATH, exist_ok=True)

# --- 1. SETUP DEVICE (GPU/CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*40}")
print(f"[INFO] PYTORCH DEVICE: {device}")
if device.type == 'cuda':
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
print(f"{'='*40}")

# --- 2. DATA LOADERS ---
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("\n[INFO] Loading Datasets...")
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

class_names = train_dataset.classes
NUM_CLASSES = len(class_names)

# --- 3. DEFINISI MODEL ---
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
            nn.Linear(256 * 14 * 14, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def get_model(model_name, num_classes):
    if model_name == 'simple_cnn':
        model = SimpleCNN(num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(weights='DEFAULT')
        for param in model.features[:-5].parameters(): param.requires_grad = False
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(weights='DEFAULT')
        for param in model.features[:-5].parameters(): param.requires_grad = False
        model.classifier[1] = nn.Linear(1280, num_classes)
    return model.to(device)

# --- 4. ENGINE TRAINING ---
def train_model(model, criterion, optimizer, model_name, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Dictionary history untuk disimpan
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n--- MULAI TRAINING: {model_name} ---")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"{model_name}.pth"))

    time_elapsed = time.time() - since
    print(f'\nTraining {model_name} selesai: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    model.load_state_dict(best_model_wts)
    return model, history

# --- 5. FUNGSI PLOTTING BARU ---
def plot_history(history, model_name, save_name):
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    epochs_range = range(len(acc)) # Otomatis menyesuaikan jumlah epoch (20)

    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend(loc='lower right')
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.legend(loc='upper right')
    
    # Simpan Gambar
    save_path = os.path.join(HISTORY_SAVE_PATH, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"[GRAPHIC] Grafik tersimpan di: {save_path}")

# --- 6. EKSEKUSI UTAMA ---
if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()

    # 1. Simple CNN
    model_cnn = get_model('simple_cnn', NUM_CLASSES)
    optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)
    _, hist_cnn = train_model(model_cnn, criterion, optimizer_cnn, 'simple_cnn_v2', EPOCHS)
    plot_history(hist_cnn, 'Simple CNN', 'simple_cnn_chart.png') # <--- Generate Grafik

    # 2. MobileNetV2
    model_mob = get_model('mobilenet', NUM_CLASSES)
    optimizer_mob = optim.Adam(model_mob.parameters(), lr=0.0001)
    _, hist_mob = train_model(model_mob, criterion, optimizer_mob, 'mobilenet_finetuned', EPOCHS)
    plot_history(hist_mob, 'MobileNetV2', 'mobilenet_chart.png') # <--- Generate Grafik

    # 3. EfficientNetB0
    model_eff = get_model('efficientnet', NUM_CLASSES)
    optimizer_eff = optim.Adam(model_eff.parameters(), lr=0.0001)
    _, hist_eff = train_model(model_eff, criterion, optimizer_eff, 'efficientnet_finetuned', EPOCHS)
    plot_history(hist_eff, 'EfficientNetB0', 'efficientnet_chart.png') # <--- Generate Grafik

    print(f"\n[SELESAI] Semua model dan grafik telah diperbarui di '{HISTORY_SAVE_PATH}'")