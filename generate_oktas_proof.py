import cv2
import matplotlib.pyplot as plt
import os

# --- KONFIGURASI PATH GAMBAR ---
# Ganti dengan path file yang valid di laptop Anda
img_path = 'static/uploads/cumulus_foto.jpg' 

if os.path.exists(img_path):
    print(f"[INFO] Memproses gambar: {img_path}")
    img = cv2.imread(img_path)
    
    # 1. Preprocessing (Sama dengan app.py)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Thresholding (Pemisahan Awan vs Langit)
    # Ambang batas 110: Piksel > 110 dianggap Awan (Putih)
    _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

    # 3. PERHITUNGAN MATEMATIS (OKTAS)
    total_pixels = img.shape[0] * img.shape[1]
    cloud_pixels = cv2.countNonZero(thresh) # Hitung piksel putih (awan)
    
    coverage_percent = (cloud_pixels / total_pixels) * 100
    oktas_val = round((coverage_percent / 100) * 8)

    print(f"[HASIL] Awan: {cloud_pixels} px | Total: {total_pixels} px")
    print(f"[HASIL] Tutupan: {coverage_percent:.2f}% -> {oktas_val}/8 OKTAS")

    # 4. VISUALISASI
    plt.figure(figsize=(12, 6))
    
    # Gambar Asli
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Citra Asli (RGB)", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Gambar Hasil Segmentasi + Teks Hasil
    plt.subplot(1, 2, 2)
    plt.imshow(thresh, cmap='gray')
    
    # Menampilkan hasil hitungan di Judul Grafik
    label_text = f"Segmentasi Biner\nEstimasi: {oktas_val}/8 OKTAS ({coverage_percent:.1f}%)"
    plt.title(label_text, fontsize=14, fontweight='bold', color='blue')
    plt.axis('off')
    
    # Simpan Bukti
    save_dir = 'static/training_history'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'oktas_proof.png')
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"[SUKSES] Bukti Visual Lengkap tersimpan di: {save_path}")
else:
    print(f"[ERROR] File gambar tidak ditemukan di: {img_path}")