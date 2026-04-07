import cv2
import torch
import numpy as np

print("\n--- Ujian Integrasi TrueSight ---")

# 1. Test OpenCV (Mata)
img = np.zeros((200, 400, 3), dtype=np.uint8)
cv2.putText(img, 'TrueSight Ready', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
print("✅ OpenCV & NumPy: Sedia untuk memproses visual.")

# 2. Test PyTorch (Otak)
x = torch.ones(5)
print(f"✅ PyTorch: Otak AI sedia dengan versi {torch.__version__}")

print("\nKESIMPULAN: Sistem anda kini mempunyai Mata dan Otak yang berfungsi!")