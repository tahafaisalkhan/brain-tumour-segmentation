# 🧠 Brain Tumor Segmentation

This project implements a deep learning-based segmentation pipeline for identifying brain tumors in MRI scans. A U-Net architecture is used to predict binary masks from grayscale brain MRI slices.

---

## 📊 Model Performance

- **Epochs Trained**: 10  
- **Final Training Loss (Epoch 10)**: `0.1830`  
- **Final Validation Loss (Epoch 10)**: `0.2752`  
- **Average Dice Coefficient on Test Set**: `0.6395`  
- **Average Intersection over Union (IoU)**: `0.5270`

---

## 🧪 Sample Predictions
![asd](https://github.com/user-attachments/assets/03daf9e5-9b7d-48f4-a43b-dbc2c9a5b45d)
![asdda](https://github.com/user-attachments/assets/08b98636-5ea5-4228-abf3-47b4aea14722)


> ⚠️ The model sometimes struggles with small or faint tumors, leading to under-segmentation.

---

## 📈 Training & Validation Loss

![Loss Curve](![output](https://github.com/user-attachments/assets/dea1dfca-8c3e-426a-92be-18e44cf21665))


Both training and validation loss decrease over time, indicating a reasonable level of convergence without signs of severe overfitting.

---

## 🏗️ Model Architecture

This project uses a custom **autoencoder-based convolutional neural network** for semantic segmentation of brain tumors from MRI scans. The architecture includes a deep encoder-decoder structure built using **PyTorch**.

### 🔧 Implementation Details

The model is defined as a class `SegmentationAE(nn.Module)` and includes the following:

#### 🧱 Encoder
- 5 convolutional blocks with increasing depth:
  - Conv2D → ReLU → MaxPool
  - Channel sizes: `1 → 32 → 64 → 128 → 256 → 512`
  - Kernel sizes: `7, 5, 3, 3, 3`
  - Gradually reduces spatial dimensions while extracting high-level features.

#### 🔁 Decoder
- 5 transposed convolutional layers:
  - ConvTranspose2D → ReLU
  - Channel sizes: `512 → 256 → 128 → 64 → 32 → 1`
  - Mirrors the encoder to reconstruct the segmentation mask.
  - Uses a final `Sigmoid` activation to produce a binary mask.

### ⚙️ Tools & Frameworks Used

- **Framework**: PyTorch  
- **Model Type**: Convolutional Autoencoder  
- **Device**: CUDA-enabled GPU or CPU fallback  
- **Input Shape**: `(1, 256, 256)` — single-channel grayscale MRI scans  
- **Output**: Binary segmentation mask of same shape  
- **Visualization**: `torchsummary` used to display model structure and parameter count

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegmentationAE().to(device)
summary(model, input_size=(1, 256, 256), batch_size=BATCH_SIZE, device=str(device))

