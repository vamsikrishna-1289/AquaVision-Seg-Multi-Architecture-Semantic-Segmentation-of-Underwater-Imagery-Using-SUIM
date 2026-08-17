# AquaVision-Seg: Multi-Architecture Semantic Segmentation of Underwater Imagery Using SUIM

AquaVision-Seg is a deep learning-based semantic segmentation project for underwater imagery using the **SUIM (Semantic Segmentation of Underwater Imagery) dataset**. The project implements and compares three modern segmentation architectures — **U-Net, DeepLabV3+, and SegFormer** — for 8-class underwater scene segmentation.

The complete pipeline covers **data preprocessing, dataset loading, model training, loss computation, evaluation, prediction, visualization, and comparative analysis**.

---

## 🚀 Project Overview

Underwater images are challenging for computer vision because of:

* Poor visibility
* Color distortion
* Low contrast
* Scattering and illumination variations
* Complex underwater objects and backgrounds

This project evaluates different semantic segmentation architectures to determine which model performs best on underwater imagery.

### Models Compared

| Model          | Pixel Accuracy |   Mean IoU |  Mean Dice |
| -------------- | -------------: | ---------: | ---------: |
| **U-Net**      |         0.7237 |     0.3547 |     0.4690 |
| **DeepLabV3+** |     **0.7280** | **0.3801** | **0.5035** |
| **SegFormer**  |         0.7153 |     0.3620 |     0.4815 |

### 🏆 Best Model

**DeepLabV3+**

* Pixel Accuracy: **72.80%**
* Mean IoU: **0.3801**
* Mean Dice: **0.5035**

DeepLabV3+ achieved the best overall performance among the three evaluated architectures.

---

## 🧠 Key Features

* 8-class semantic segmentation
* SUIM underwater image dataset
* U-Net implementation
* DeepLabV3+ implementation
* SegFormer implementation
* Image and mask preprocessing
* Cross-Entropy + Dice Loss
* Pixel Accuracy evaluation
* Mean IoU evaluation
* Mean Dice evaluation
* Per-class IoU and Dice analysis
* Model checkpointing
* Test-set prediction
* Ground-truth visualization
* Prediction visualization
* Model comparison
* Quantitative and qualitative analysis

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────────┐
                    │      SUIM Dataset       │
                    │  Underwater Images +    │
                    │      Ground Truth       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Preprocessing       │
                    │                         │
                    │ • Resize to 256 × 256   │
                    │ • Image normalization   │
                    │ • Mask preprocessing    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      SUIM Dataset       │
                    │      DataLoader         │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
        ┌───────────┐      ┌──────────────┐   ┌───────────┐
        │  U-Net    │      │ DeepLabV3+   │   │ SegFormer │
        └─────┬─────┘      └──────┬───────┘   └─────┬─────┘
              │                   │                  │
              └───────────────────┼──────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │    Model Predictions    │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
                 ▼               ▼               ▼
          ┌────────────┐  ┌─────────────┐  ┌────────────┐
          │ Pixel Acc. │  │   Mean IoU  │  │ Mean Dice  │
          └────────────┘  └─────────────┘  └────────────┘
                 │               │               │
                 └───────────────┼───────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Model Comparison      │
                    │                         │
                    │      🏆 DeepLabV3+      │
                    └─────────────────────────┘
```

---

## 📊 Dataset

The project uses the **SUIM — Semantic Segmentation of Underwater Imagery** dataset.

The dataset contains underwater images with pixel-level segmentation masks.

### Dataset Configuration

* Total image-mask pairs: **1,525**
* Training samples: **1,220**
* Validation samples: **305**
* Image size: **256 × 256**
* Input channels: **3 RGB**
* Number of classes: **8**

### Segmentation Classes

| Class ID | Class               |
| -------: | ------------------- |
|        0 | Background          |
|        1 | Human Diver         |
|        2 | Aquatic Plants      |
|        3 | Wrecks/Ruins        |
|        4 | Robots              |
|        5 | Reefs/Invertebrates |
|        6 | Fish/Vertebrates    |
|        7 | Sea Floor/Rocks     |

---

## 🛠️ Technologies

* **Python**
* **PyTorch**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **U-Net**
* **DeepLabV3+**
* **SegFormer**
* **SUIM Dataset**
* **Semantic Segmentation**

---

## 📁 Project Structure

```text
Project 1/
│
├── SUIM/
│   └── train_val/
│       ├── images/
│       └── masks/
│
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── visualization.py
│   │
│   └── models/
│       ├── unet.py
│       ├── deeplabv3plus.py
│       └── segformer.py
│
├── scripts/
│   ├── train_unet.py
│   ├── train_deeplabv3plus.py
│   ├── train_segformer.py
│   ├── evaluate.py
│   ├── predict.py
│   └── analyze_results.py
│
├── outputs/
│   ├── checkpoints/
│   │   ├── unet_best.pth
│   │   ├── deeplabv3plus_best.pth
│   │   └── segformer_best.pth
│   │
│   ├── predictions/
│   │   ├── unet/
│   │   ├── deeplabv3plus/
│   │   └── segformer/
│   │
│   ├── visualizations/
│   └── analysis/
│
├── results/
│   ├── metrics.csv
│   └── model_comparison.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Project-1
```

## 2. Create a virtual environment

```bash
python3 -m venv .venv
```

## 3. Activate the virtual environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Dependencies

The project uses the following major libraries:

```text
torch
torchvision
numpy
pandas
matplotlib
Pillow
```

---

# 🔄 Data Preprocessing

The preprocessing pipeline performs:

1. Image loading
2. Image resizing to **256 × 256**
3. Image normalization
4. Ground-truth mask resizing
5. Conversion to PyTorch tensors
6. Preparation of segmentation labels

Example dataset verification:

```bash
python -c "from src.dataset import SUIMDataset; d=SUIMDataset('SUIM/train_val/images','SUIM/train_val/masks',training=True); image, mask=d[0]; print(image.shape, mask.shape)"
```

Expected output:

```text
torch.Size([3, 256, 256])
torch.Size([256, 256])
```

---

# 🧮 Loss Function

The project uses a combined **Cross-Entropy + Dice Loss**.

```text
Total Loss = CE Loss + Dice Loss
```

### Cross-Entropy Loss

Handles pixel-level classification.

### Dice Loss

Encourages better overlap between predicted segmentation regions and ground-truth regions.

This combination is useful for segmentation problems where different classes can have significantly different pixel distributions.

Test the loss implementation:

```bash
python src/losses.py
```

---

# 🧠 Model Architectures

## 1. U-Net

U-Net is an encoder-decoder architecture designed specifically for semantic segmentation.

It uses:

* Encoder
* Bottleneck
* Decoder
* Skip connections

The skip connections help preserve spatial information from earlier layers.

---

## 2. DeepLabV3+

DeepLabV3+ combines:

* Atrous convolution
* Atrous Spatial Pyramid Pooling
* Encoder-decoder architecture
* Multi-scale contextual information

It performed best among the three models in this project.

---

## 3. SegFormer

SegFormer uses a Transformer-based architecture for semantic segmentation.

It combines:

* Hierarchical Transformer encoder
* Lightweight MLP decoder
* Multi-scale feature representations

SegFormer provides a modern Transformer-based alternative to convolutional segmentation architectures.

---

# 🏋️ Training

Each architecture was trained using the same general segmentation pipeline.

### Training Configuration

```text
Batch Size      : 4
Epochs          : 20
Learning Rate   : 0.0001
Input Size      : 256 × 256
Classes         : 8
Device          : Apple Silicon MPS
Loss            : Cross-Entropy + Dice
```

---

## Train U-Net

```bash
python scripts/train_unet.py
```

Checkpoint:

```text
outputs/checkpoints/unet_best.pth
```

---

## Train DeepLabV3+

```bash
python scripts/train_deeplabv3plus.py
```

Checkpoint:

```text
outputs/checkpoints/deeplabv3plus_best.pth
```

---

## Train SegFormer

```bash
python scripts/train_segformer.py
```

Checkpoint:

```text
outputs/checkpoints/segformer_best.pth
```

---

# 📈 Evaluation Metrics

The project evaluates segmentation performance using:

### Pixel Accuracy

Measures the percentage of correctly classified pixels.

```text
Pixel Accuracy =
Correctly Classified Pixels / Total Pixels
```

### Mean IoU

Intersection over Union averaged across segmentation classes.

```text
IoU = Intersection / Union
```

### Mean Dice

Measures overlap between prediction and ground truth.

```text
Dice = 2 × Intersection / (Prediction + Ground Truth)
```

Per-class IoU and Dice scores are also calculated.

---

# 🔍 Model Evaluation

The trained models can be evaluated using:

```bash
python scripts/evaluate.py
```

The evaluation results are stored in:

```text
results/
├── metrics.csv
└── model_comparison.csv
```

---

# 🖼️ Prediction

Predictions can be generated using:

```bash
python scripts/predict.py
```

Generated predictions are stored under:

```text
outputs/predictions/
```

with separate directories for:

```text
unet/
deeplabv3plus/
segformer/
```

---

# 🎨 Visualization

The project generates qualitative visualizations comparing:

* Original underwater image
* Ground-truth segmentation
* Model prediction
* Model overlays

Example output:

```text
outputs/visualizations/
```

The visualization pipeline allows qualitative comparison between the three architectures.

---

# 📊 Results Analysis

The final model comparison is generated using:

```bash
python scripts/analyze_results.py
```

Generated analysis files include:

```text
outputs/analysis/
├── pixel_accuracy_comparison.png
├── mean_iou_comparison.png
├── mean_dice_comparison.png
├── model_comparison.png
└── per_class_iou_comparison.png
```

---

# 🏆 Final Results

| Metric         |  U-Net | DeepLabV3+ | SegFormer |
| -------------- | -----: | ---------: | --------: |
| Pixel Accuracy | 0.7237 | **0.7280** |    0.7153 |
| Mean IoU       | 0.3547 | **0.3801** |    0.3620 |
| Mean Dice      | 0.4690 | **0.5035** |    0.4815 |

### Best Overall Model: DeepLabV3+

DeepLabV3+ achieved the highest:

* Pixel Accuracy
* Mean IoU
* Mean Dice

Therefore, based on the evaluated metrics, **DeepLabV3+ is the best-performing architecture in this experiment**.

---

# 📊 Per-Class IoU

| Class               |      U-Net | DeepLabV3+ |  SegFormer |
| ------------------- | ---------: | ---------: | ---------: |
| Background          |     0.7850 |     0.7666 |     0.7644 |
| Human Diver         |     0.3741 | **0.4763** |     0.4431 |
| Aquatic Plants      |     0.0000 |     0.0313 | **0.0467** |
| Wrecks/Ruins        |     0.3366 |     0.3815 | **0.3859** |
| Robots              |     0.0000 | **0.0371** |     0.0000 |
| Reefs/Invertebrates | **0.6158** |     0.6126 |     0.6001 |
| Fish/Vertebrates    | **0.3208** |     0.3115 |     0.2427 |
| Sea Floor/Rocks     |     0.4052 | **0.4242** |     0.4131 |

The results show that model performance varies significantly across semantic categories. DeepLabV3+ provides the strongest overall balance across the evaluated classes.

---

# 🔬 Qualitative Comparison

The project also provides visual comparisons of predictions from all three models.

For each test image, the generated visualization can be used to compare:

```text
Original Image
      ↓
Ground Truth
      ↓
U-Net Prediction
      ↓
DeepLabV3+ Prediction
      ↓
SegFormer Prediction
```

These visualizations complement the quantitative metrics and help assess how well each architecture captures underwater objects and regions.

---

# 💾 Model Checkpoints

The trained model weights are saved as:

```text
outputs/checkpoints/
├── unet_best.pth
├── deeplabv3plus_best.pth
└── segformer_best.pth
```

These checkpoints can be used for:

* Evaluation
* Inference
* Visualization
* Future fine-tuning

**Note:** Model checkpoint files can be several hundred MB or more. They generally should **not be committed directly to GitHub** if they make the repository excessively large. Use `.gitignore` for local checkpoints and optionally provide a separate model-download location.

---

# 📁 Output Organization

```text
outputs/
│
├── checkpoints/
│   ├── unet_best.pth
│   ├── deeplabv3plus_best.pth
│   └── segformer_best.pth
│
├── predictions/
│   ├── unet/
│   ├── deeplabv3plus/
│   └── segformer/
│
├── visualizations/
│
└── analysis/
    ├── pixel_accuracy_comparison.png
    ├── mean_iou_comparison.png
    ├── mean_dice_comparison.png
    ├── model_comparison.png
    └── per_class_iou_comparison.png
```

---

# 🚀 Complete Workflow

```text
1. Dataset
      ↓
2. Preprocessing
      ↓
3. Dataset & DataLoader
      ↓
4. Train U-Net
      ↓
5. Train DeepLabV3+
      ↓
6. Train SegFormer
      ↓
7. Save Best Checkpoints
      ↓
8. Evaluate Models
      ↓
9. Generate Predictions
      ↓
10. Visualize Results
      ↓
11. Compare Metrics
      ↓
12. Select Best Model
```

---

# 🎯 Applications

The developed segmentation pipeline can support underwater computer vision applications such as:

* Marine ecosystem monitoring
* Underwater robotics
* Autonomous underwater vehicles
* Marine object detection
* Coral reef monitoring
* Underwater exploration
* Aquatic environment analysis
* Subsea inspection

---

# 🔮 Future Improvements

Potential extensions include:

* Data augmentation
* Class-balanced loss functions
* Learning-rate scheduling
* Longer training schedules
* Hyperparameter optimization
* Transfer learning
* Stronger Transformer-based segmentation models
* Improved handling of minority classes
* Underwater image enhancement before segmentation
* Real-time inference optimization
* Deployment on underwater robotic platforms

---

# 📌 Conclusion

AquaVision-Seg implements a complete semantic segmentation pipeline for underwater imagery using the SUIM dataset. Three architectures — **U-Net, DeepLabV3+, and SegFormer** — were trained and evaluated under the same experimental framework.

Among the evaluated models, **DeepLabV3+ achieved the best overall performance**, obtaining a **Mean IoU of 0.3801**, **Mean Dice of 0.5035**, and **Pixel Accuracy of 0.7280**.

The project demonstrates how different segmentation architectures perform on challenging underwater imagery and provides both quantitative metrics and qualitative visualizations for model comparison.

---

## 👨‍💻 Author

**Vamsi Krishna Gondu**

Master of Science — Artificial Intelligence and Intelligent Systems

---

## 📜 License

This project is intended for **academic and research purposes**. Please refer to the SUIM dataset's original licensing and usage terms before redistributing the dataset.

---

## ⭐ Acknowledgements

This project uses the **SUIM underwater semantic segmentation dataset** and builds upon established semantic segmentation architectures including U-Net, DeepLabV3+, and SegFormer.
