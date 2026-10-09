# 🕐 TIMEKEEPER 2.0 - Historical Photo Restoration & Enhancement

> Bringing Old Photos Back to Life with Advanced AI-Powered Image Restoration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)

---

## 📸 Project Overview

**TIMEKEEPER 2.0** is a comprehensive solution for restoring and enhancing old, damaged photographs. Using cutting-edge deep learning techniques, it combines multiple AI models to automatically detect and remove scratches, enhance faces, colorize black-and-white images, and restore overall photo quality.

### Key Features

- ✨ **Automatic Scratch Detection & Removal** - Detect and intelligently remove physical damage
- 🎨 **AI-Powered Colorization** - Realistic color restoration using DDColor (ICCV 2023)
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
│  2. DDColor Colorization │
│  (Optional: Add Vivid    │
│   Realistic Colors)      │
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

## 🖼️ Results Preview

Below you can see the restoration pipeline in action with before and after comparisons, along with various output formats:

![TIMEKEEPER UI Screenshot](https://github.com/Nihir-Soni/TIMEKEEPER-2.0/raw/main/imgs/ui_screenshot.png)

**Example Outputs:**
- **Structural Restoration** - Removes scratches and major damage
- **AI Colorization** - Adds realistic, vibrant colors to B&W photos
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

**Standalone colorization:**
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
- `--colorize` - Enable DDColor colorization
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

4. **Colorization Module** (DDColor)
   - ICCV 2023 state-of-the-art
   - Realistic color restoration
   - User-guided optional mode

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

If you use TIMEKEEPER 2.0 in your research, please cite:

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

@inproceedings{zhang2023ddcolor,
  title={DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders},
  author={Zhang, Xiaozhong and Wang, Xiuping and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
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
- **[DDColor Repository](https://github.com/piddnad/DDColor)**

---

## 📧 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/Nihir-Soni/TIMEKEEPER-2.0/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Nihir-Soni/TIMEKEEPER-2.0/discussions)
- **Email**: Contact repository maintainer

---

## 🙏 Acknowledgments

- Original research by [Ziyu Wan](http://raywzy.com/) et al., Microsoft Research Asia
- DDColor implementation by [Xiaozhong Zhang](https://github.com/piddnad/DDColor)
- Community contributions and testing

---

<div align="center">

**Made with ❤️ for preserving memories**

⭐ If you find this project helpful, please consider giving it a star!

</div>
