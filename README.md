# 🕐 TIMEKEEPER - Historical Photo Restoration & Enhancement

> Bringing Old Photos Back to Life with Advanced AI-Powered Image Restoration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)

---

## 📸 Project Overview

**TIMEKEEPER** is a comprehensive solution for restoring and enhancing old, damaged photographs. Using deep learning, it combines multiple image restoration stages to automatically repair scratches, improve facial details, and restore color and quality in historical images.

### Key Features

- ✨ **Automatic Scratch Detection & Removal** - Detect and intelligently remove physical damage
- 🎨 **AI-Powered Colorization** - Realistic color restoration using the SIGGRAPH 2017 colorization model
- 👤 **Progressive Face Enhancement** - High-quality facial region refinement
- 🖼️ **Global Image Restoration** - Comprehensive restoration for structured and unstructured degradation
- 🔍 **Uncertainty Estimation** - Confidence maps for restoration reliability
- 🖥️ **User-Friendly GUI** - Desktop application for easy photo restoration
- 🌐 **Web UI & API Support** - RESTful API and web interface for integration
- 🐳 **Docker & Kubernetes Support** - Production-ready containerization
- 📊 **High-Resolution Support** - Enhanced processing for large images

---

## 🎯 Restoration Pipeline

```
┌──────────────┐
│ Input Image  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│  1. Global Restoration   │
│  (Scratch Detection &    │
│   Quality Enhancement)   │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│  2. SIGGRAPH 2017        │
│  Colorization Model       │
│  (Realistic grayscale -> │
│   color restoration)     │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│  3. Face Enhancement     │
│  (Progressive Generator  │
│   for Facial Refinement) │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│  Final Output Image      │
│  (Fully Restored Photo)  │
└──────────────────────────┘
```

---

## 🖼️ Live Web Interface Demo

Here's how the complete TIMEKEEPER web interface looks and works in practice:

<p align="center">
  <img src="https://raw.githubusercontent.com/Nihir-Soni/TIMEKEEPER-2.0/main/test_images/ui.png" alt="TIMEKEEPER Web Interface - Before and After Comparison with Restoration Analysis" width="100%" />
</p>

**Web Interface Features:**
- 📷 **Photograph Input** - Upload historical photos for restoration
- 🔄 **Before & After Comparison** - Side-by-side visualization of restoration process
- 🎛️ **Restoration Options** - Toggle scratch removal, colorization, and high-resolution mode
- 📊 **Restoration Analysis** - View all intermediate stages (Structural Restoration, AI Colorization, Uncertainty Maps, Confidence Scores)
- 🎨 **Color Saturation Control** - Adjust color intensity from desaturated to vivid
- ✅ **Multiple Output Formats** - Restored image, colorized version, uncertainty maps, and confidence calibration

---

## 🖼️ Results Preview

Below are sample input and output images already included in the repository. The colorization stage is based on the SIGGRAPH 2017 model, not DDColor.

### Example Gallery

<p align="center">
  <img src="https://raw.githubusercontent.com/Nihir-Soni/TIMEKEEPER-2.0/main/test_images/old/ab.png" width="300" alt="Input old photo B" />
  <img src="https://raw.githubusercontent.com/Nihir-Soni/TIMEKEEPER-2.0/main/test_output/final_output/a.png" width="300" alt="Restored output A" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Nihir-Soni/TIMEKEEPER-2.0/main/test_images/old/c.png" width="300" alt="Input old photo C" />
  <img src="https://raw.githubusercontent.com/Nihir-Soni/TIMEKEEPER-2.0/main/test_images/old/b.png" width="300" alt="Scratch-restored output A" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Nihir-Soni/TIMEKEEPER-2.0/main/test_images/old_w_scratch/a.png" width="300" alt="Damaged input image" />
  <img src="https://raw.githubusercontent.com/Nihir-Soni/TIMEKEEPER-2.0/main/test_scratch_output/final_output/a.png" width="300" alt="Damaged output restored" />
</p>

**Example Outputs:**
- **Structural Restoration** - Removes scratches and major damage
- **SIGGRAPH 2017 Colorization** - Adds realistic, vibrant colors to grayscale archival images
- **Raw Uncertainty Map** - Confidence visualization of restoration
- **Calibrated Confidence** - Reliability metrics for each region

---

## 🚀 Quick Start

### Prerequisites

- Python >= 3.6
- CUDA 11.0+ (for GPU acceleration)
- 8GB+ RAM (16GB+ recommended)
- ~5GB disk space for models

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Nihir-Soni/TIMEKEEPER-2.0.git
cd TIMEKEEPER-2.0
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download pre-trained models:**
```bash
python download_and_extract.py
```

Or manually download:
- Face detection landmarks: [dlib predictor](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- Face enhancement models: [GitHub releases](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases)
- Global restoration models: [GitHub releases](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases)

4. **Download colorization checkpoint (optional):**
```bash
python Colorization/download_model.py
```

---

## 💻 Usage

### 1️⃣ Full Pipeline (Recommended)

**For clean images without scratches:**
```bash
python run.py --input_folder ./test_images \
              --output_folder ./output \
              --GPU 0
```

**For scratched/damaged images:**
```bash
python run.py --input_folder ./test_images \
              --output_folder ./output \
              --GPU 0 \
              --with_scratch
```

**For high-resolution images with scratches:**
```bash
python run.py --input_folder ./test_images \
              --output_folder ./output \
              --GPU 0 \
              --with_scratch \
              --HR
```

### 2️⃣ GUI Application

Interactive desktop application for single image processing:

```bash
python GUI.py
```

**Steps:**
1. Launch the application
2. Click "Browse" and select a photo
3. Check desired options (Remove Scratches, Colorize, etc.)
4. Click "Restore Photo"
5. View results in the GUI window
6. Save to output folder

### 3️⃣ Web API

Start the REST API server:
```bash
python api.py
```

Example request:
```bash
curl -X POST http://localhost:5000/restore \
  -F "file=@photo.jpg" \
  -F "with_scratch=true" \
  -F "colorize=true"
```

### 4️⃣ Individual Components

**Scratch Detection Only:**
```bash
cd Global/
python detection.py --test_path ./test_images \
                    --output_dir ./output \
                    --input_size full_size
```

**Global Restoration Only:**
```bash
cd Global/
python test.py --Scratch_and_Quality_restore \
               --test_input ./test_images \
               --test_mask ./masks \
               --outputs_dir ./output
```

**Face Enhancement Only:**
```bash
cd Face_Enhancement/
python test.py --test_input ./cropped_faces \
               --outputs_dir ./output
```

### 5️⃣ Colorization

**Standalone colorization using the SIGGRAPH 2017 pipeline:**
```bash
python Colorization/demo_release.py -i input_image.jpg -o output_image.jpg
```

---

## 📊 Output Structure

```
output/
├── final_output/              # Final restored images
├── restored/                  # After global restoration
├── colorized/                 # After colorization (if enabled)
├── face_enhanced/             # After face enhancement
├── uncertainty_map/           # Confidence maps
└── detection_results/         # Scratch detection masks (if enabled)
```

---

## 🛠️ Configuration & Advanced Options

### GPU Support
- Single GPU: `--GPU 0`
- Multiple GPUs: `--GPU 0,1,2`
- CPU only: `--GPU -1` (slow)

### Image Size Handling
- `full_size` - Process at original resolution
- `resize_256` - Resize to 256×256 (faster)
- `scale_256` - Smart scaling maintaining aspect ratio

### Restoration Options
- `--with_scratch` - Enable scratch detection and removal
- `--HR` - High-resolution mode (slower but better quality)
- `--colorize` - Enable SIGGRAPH 2017 colorization
- `--enhancement_ratio 1.0` - Face enhancement intensity (0.0-2.0)

---

## 🐳 Docker & Kubernetes

### Docker

```bash
# Build image
docker build -t timekeeper:latest .

# Run container
docker run --gpus all -v $(pwd)/test_images:/app/input \
           -v $(pwd)/output:/app/output \
           timekeeper:latest python run.py --input_folder /app/input \
           --output_folder /app/output --GPU 0
```

### Kubernetes

```bash
kubectl apply -f kubernetes-pod.yml
```

### Ansible Deployment

```bash
ansible-playbook ansible.yaml
```

---

## 🔬 Technical Details

### Architecture Components

1. **Global Restoration Module**
   - Triplet domain translation network
   - Handles structured and unstructured degradation
   - VAE-based approach for quality enhancement

2. **Scratch Detection Module**
   - Deep learning-based detection
   - Generates mask for damaged regions
   - Automated preprocessing

3. **Face Enhancement Module**
   - Progressive generator architecture
   - Face-specific quality refinement
   - Detail preservation

4. **Colorization Module** (SIGGRAPH 2017)
   - Real-time user-guided image colorization with learned deep priors
   - Produces natural and realistic restoration for grayscale archival pictures
   - Compatible with the repository's `Colorization` implementation

### Model Specifications

| Component | Input Size | Model Size | GPU Memory |
|-----------|-----------|-----------|-----------|
| Global Restoration | 256×256 | ~180MB | 2GB |
| Face Enhancement | 256×256 | ~150MB | 1.5GB |
| Scratch Detection | Variable | ~80MB | 1GB |
| Colorization | 256×256 | ~280MB | 1.5GB |

---

## 📦 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.19.0
Pillow>=8.0.0
scikit-image>=0.18.0
scipy>=1.5.0
tensorboard>=2.4.0
dlib>=19.20
```

See `requirements.txt` for complete dependencies.

---

## 🎓 Training (Advanced)

For custom model training, see the [Training Guide](./research/TRAINING.md):

```bash
# Prepare dataset
cd Global/data/
python Create_Bigfile.py

# Train VAE domain models
python train_domain_A.py --training_dataset domain_A --dataroot ./data

# Train mapping network
python train_mapping.py --training_dataset mapping --dataroot ./data
```

---

## 📚 References & Citations

If you use TIMEKEEPER in your research, please cite:

```bibtex
@inproceedings{wan2020bringing,
  title={Bringing Old Photos Back to Life},
  author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2747--2757},
  year={2020}
}

@article{wan2022old,
  title={Old Photo Restoration via Deep Latent Space Translation},
  author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={12},
  pages={9114--9129},
  year={2022},
  publisher={IEEE}
}

@article{zhang2017real,
  title={Real-Time User-Guided Image Colorization with Learned Deep Priors},
  author={Zhang, Richard and Zhu, Jun-Yan and Isola, Phillip and Geng, Xinyang and Lin, Angela S and Yu, Tianhe and Efros, Alexei A},
  journal={ACM Transactions on Graphics (TOG)},
  volume={36},
  number={4},
  year={2017},
  publisher={ACM}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) for details.

---

## ⚠️ Security & Safety

For security concerns, please refer to [SECURITY.md](./SECURITY.md).

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](./LICENSE) file for details.

**Note:** This project includes research code and has not been optimized for production deployment. Model architectures and weights are provided for academic and non-commercial use.

---

## 🔗 Project Links

- **[Project Website](http://raywzy.com/Old_Photo/)**
- **[CVPR 2020 Paper](https://arxiv.org/abs/2004.09484)**
- **[TPAMI 2022 Paper](https://arxiv.org/pdf/2009.07047v1.pdf)**
- **[Original Repository](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)**
- **[SIGGRAPH 2017 Colorization Repository](https://github.com/richzhang/colorization)**

---

## 📧 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/Nihir-Soni/TIMEKEEPER-2.0/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Nihir-Soni/TIMEKEEPER-2.0/discussions)
- **Email**: Contact repository maintainer

---

## 🙏 Acknowledgments

- Original research by [Ziyu Wan](http://raywzy.com/) et al., Microsoft Research Asia
- SIGGRAPH 2017 colorization work by [Richard Zhang](https://richzhang.github.io/) and collaborators
- Community contributions and testing

---

<div align="center">

**Made with ❤️ for preserving memories**

⭐ If you find this project helpful, please consider giving it a star!

</div>
