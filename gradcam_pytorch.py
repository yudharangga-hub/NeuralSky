import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook untuk menangkap gradien & aktivasi
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # 1. Forward Pass
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output)

        # 2. Backward Pass (Hitung Gradien)
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # 3. Generate Heatmap
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling pada gradien (b, c, h, w) -> (b, c, 1, 1)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Kalikan weight dengan activation map
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU & Normalisasi
        cam = F.relu(cam)
        cam = cam.view(cam.size(2), cam.size(3)).detach().cpu().numpy()
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8) # Avoid zero div
        
        return cam

def save_gradcam(img_path, heatmap, save_path):
    # Baca gambar asli
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    # Resize heatmap agar sesuai gambar
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Ubah ke format warna (JET)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Gabungkan (Overlay)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(save_path, superimposed_img)