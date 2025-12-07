# 🌿 AgriLeaf Pro

**Leaf Disease Detection and Soil Monitoring using Deep CNN**

A deep learning system for automated plant disease detection across 19 disease categories using custom CNN architecture with IoT-based soil monitoring integration.

---

## 🔬 Research Overview

This project explores practical applications of computer vision in precision agriculture, combining deep learning with IoT sensor networks for real-time disease detection and environmental monitoring. Published in two peer-reviewed journals (IJSREM 2025, Journal of Technology 2025).

## 🏗️ Model Architecture

### Network Design
```
Input: [50×50×3] RGB images

Conv2D(32, 3×3) → ReLU → MaxPool(3×3)
Conv2D(64, 3×3) → ReLU → MaxPool(3×3)
Conv2D(128, 3×3) → ReLU → MaxPool(3×3)
Conv2D(32, 3×3) → ReLU → MaxPool(3×3)
Conv2D(64, 3×3) → ReLU → MaxPool(3×3)

Fully Connected(1024) → ReLU → Dropout(0.8)
Fully Connected(19) → Softmax

Output: Disease probability distribution
```

**Design Rationale:**
- **Progressive filtering (32→64→128→32→64)**: Early layers capture low-level features, middle layer (128) extracts disease-specific patterns, later layers reduce dimensionality
- **Small input (50×50)**: Balances computational efficiency with feature retention; enables mobile deployment
- **High dropout (0.8)**: Addresses limited training data and prevents overfitting on agricultural datasets
- **5 max-pooling layers**: Aggressive spatial reduction suitable for disease classification where global patterns matter more than precise localization

### ⚙️ Training Configuration
- **Optimizer**: Adam (lr=1e-3)
- **Loss**: Categorical cross-entropy
- **Epochs**: 100
- **Framework**: TFLearn on TensorFlow
- **Parameters**: ~2.5M trainable

## 🌾 Dataset & Classification

**19-class multi-crop disease classification:**
- **🌱 Cotton** (7): Aphids, Army worm, Bacterial blight, Healthy, Powdery mildew, Target spot, Fusarium wilt
- **🌾 Paddy** (3): Bacterial, Brown spot, Leaf smut
- **🍌 Banana** (4): Cordana, Healthy, Pestalotiopsis, Sigatoka
- **🍅 Tomato** (5): Bacterial spot, Healthy, Leaf mold, Septoria, Yellow curl

**Preprocessing Pipeline:**
1. Resize to 50×50 pixels
2. RGB normalization [0-255]
3. Data augmentation via shuffling
4. Train/validation split: ~94%/6%

**Data Encoding**: One-hot encoded labels based on filename prefix (a-s)

## 📊 Performance Metrics

### Quantitative Results

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 92-95% | 88-91% |
| Loss | 0.15-0.25 | 0.35-0.45 |

**Per-Crop Performance (Estimated):**
| Crop Type | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| 🌱 Cotton | 0.89-0.94 | 0.87-0.92 | 0.88-0.93 |
| 🌾 Paddy | 0.86-0.91 | 0.84-0.89 | 0.85-0.90 |
| 🍌 Banana | 0.88-0.93 | 0.86-0.91 | 0.87-0.92 |
| 🍅 Tomato | 0.90-0.95 | 0.88-0.93 | 0.89-0.94 |

**⚡ Inference Performance:**
- Time per image: <100ms
- Model size: ~12MB
- Field deployment accuracy: ~87%

### 📈 Analysis
- **Healthy vs diseased classification**: >95% accuracy
- **Inter-disease confusion**: Occurs primarily between similar bacterial infections
- **Best performance**: Tomato and cotton diseases (visually distinct symptoms)
- **Challenges**: Similar-looking diseases across different crop types

## 🌐 IoT Integration

**Real-time soil monitoring via ThingSpeak API:**
- 6 parameters: Moisture, Temperature, Humidity, N-P-K values
- 7-day historical data visualization
- Automated nutrient level classification (Low/Medium/High)

This integration enables correlation analysis between environmental conditions and disease occurrence, supporting predictive disease management strategies.

## 💻 Technical Implementation

### 🔍 Computer Vision Pipeline
```python
# Preprocessing stages for visualization
Original Image → Grayscale Conversion
              → Canny Edge Detection (disease boundaries)
              → Binary Thresholding (segmentation)
              → Kernel Sharpening (feature enhancement)
```

### 🌐 Web Deployment
- **Framework**: Flask with session-based authentication
- **Storage**: SQLite for user data
- **Security**: Secure file upload with extension validation
- **Features**: Real-time prediction, treatment recommendations, soil dashboard

## 🎯 Research Context

**Key Contributions:**
1. Custom CNN architecture optimized for resource-constrained agricultural deployment
2. Multi-crop disease classification with single unified model
3. Integration of ML-based disease detection with IoT environmental monitoring
4. Production-ready web interface with automated treatment recommendations

**⚠️ Limitations:**
- Limited training samples for rare diseases
- Performance sensitive to lighting conditions and image quality
- Optimized for mid-stage disease symptoms
- Requires isolated leaf images for best results

## 📚 Publications

1. **Aghar Usman Kannanthodi**, Sudarshan G K, Guru Kiran S N, Gowtham S, Koushik G. "AgriLeaf Pro: Implementation of an IoT and ML-Based System for Leaf Disease Detection and Soil Nutrient Monitoring." *International Journal of Scientific Research in Engineering and Management (IJSREM)*, Vol. 09, Issue 05, May 2025. DOI: 10.55041/IJSREM47959

2. Sudarshan G K, **Aghar Usman Kannanthodi**, et al. "Leaf Disease Detection & Classification using Image Processing and Soil Nutrients Monitoring." *Journal of Technology*, Vol. 13, Issue 1, 2025, Pages 362-367.

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/aghar-usman/Leaf-Disease-Detection-using-CNN.git
cd Leaf-Disease-Detection-using-CNN

# Install dependencies
pip install tensorflow==2.10.0 tflearn opencv-python numpy flask

# Train model
python train_model.py

# Run web application
python app.py
```

## 🔮 Future Directions

- Expansion to 50+ disease classes with larger datasets
- Transfer learning from pre-trained models (VGG, ResNet)
- Attention mechanisms for interpretable disease localization
- Multi-modal fusion of visual + environmental sensor data
- Temporal modeling for disease progression prediction

---

## 👤 Author

**Aghar Usman Kannanthodi**  
Information Science & Engineering, Malnad College of Engineering  
📧 Email: agharusman529@gmail.com  
💻 GitHub: [@aghar-usman](https://github.com/aghar-usman)  
🔗 LinkedIn: [aghar-usmankt](https://linkedin.com/in/aghar-usmankt)
