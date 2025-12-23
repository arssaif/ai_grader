# 🏥 AI Grader For Radiologists

A comprehensive AI-powered web application for automated chest X-ray analysis, providing multi-modal diagnostic support including disease classification, image segmentation, opacity detection, and automated report generation.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [AI Models](#-ai-models)
- [API Documentation](#-api-documentation)
- [Technologies Used](#-technologies-used)
- [Youtube Demo](#-youtube-demo)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🔬 Diagnostic Capabilities
- **Disease Classification**: Multi-label classification for 12 thoracic pathologies
  - Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule
  - Pneumothorax, Consolidation, Edema, Emphysema, Pleural Thickening, No Finding

- **Automated Image Captioning**: Natural language report generation for X-ray findings

- **Anatomical Segmentation**: Precise segmentation of:
  - Lungs (left and right)
  - Heart
  - Clavicles
  - Cardio-Thoracic Ratio (CTR) calculation

- **Grad-CAM Heatmaps**: Visual attention maps showing regions of interest for disease prediction

- **Opacity Detection**: Automated detection of lung opacities using YOLOv5

- **External Device Detection**: Detection of cardiac devices and medical implants

### 👥 User Management
- Role-based access control (Admin/Doctor)
- Secure authentication and session management
- Patient image management and organization
- Multi-user support with isolated data storage

### 📊 Reporting
- Comprehensive diagnostic reports with all analysis results
- Visual presentation of findings with heatmaps and segmentations
- Downloadable reports for clinical documentation

## 🏗️ System Architecture

```
┌─────────────┐
│   Frontend  │  (HTML, CSS, JavaScript)
│  Templates  │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│    Flask    │  (REST API, Session Management)
│   Backend   │
└──────┬──────┘
       │
       ├────────────────┬──────────────┬──────────────┬──────────────┐
       ↓                ↓              ↓              ↓              ↓
┌──────────┐   ┌────────────┐  ┌─────────────┐  ┌──────────┐  ┌──────────┐
│  Disease │   │   Image    │  │Segmentation │  │ Opacity  │  │ External │
│Classifier│   │ Captioning │  │   Models    │  │ Detector │  │ Devices  │
│(EfficNet)│   │(Encoder-   │  │   (U-Net)   │  │ (YOLOv5) │  │ (YOLOv5) │
│          │   │ Decoder)   │  │             │  │          │  │          │
└──────────┘   └────────────┘  └─────────────┘  └──────────┘  └──────────┘
       │                │              │              │              │
       └────────────────┴──────────────┴──────────────┴──────────────┘
                                       │
                                       ↓
                              ┌────────────────┐
                              │   SQLite DB    │
                              │  (User Data,   │
                              │ Image Metadata)│
                              └────────────────┘
```

## 📁 Project Structure

```
FYP_WebApp/
│
├── ai_grader/                      # AI/ML inference modules
│   ├── perform_classification.py   # Disease classification
│   ├── generate_caption.py         # Report generation
│   ├── generate_segmentation.py    # Lung/heart segmentation
│   ├── generate_heatmaps.py        # Grad-CAM visualization
│   ├── detect_opacity.py           # Opacity detection
│   └── detect_external_devices.py  # Device detection
│
├── db_src/                         # Database models
│   └── DB_MODEL.py                 # SQLAlchemy models
│
├── routes/                         # API route definitions
│   └── authRoutes.py               # Authentication routes
│
├── views/                          # API view handlers
│   └── AuthViews.py                # Auth API endpoints
│
├── templates/                      # HTML templates
│   ├── home.html                   # Main dashboard
│   ├── login.html                  # Login page
│   ├── signup.html                 # Registration page
│   ├── AdminHome.html              # Admin panel
│   ├── about.html                  # About page
│   ├── contact.html                # Contact page
│   └── welcome.html                # Landing page
│
├── static/                         # Static assets
│   ├── css_files/                  # Stylesheets
│   ├── image/                      # UI images
│   ├── models/                     # Pre-trained model weights
│   │   ├── disease_classification/ # EfficientNet models
│   │   ├── captioning/             # Encoder-Decoder models
│   │   ├── segmentation/           # U-Net models
│   │   ├── heatmap/                # Heatmap models
│   │   ├── opacity/                # YOLO opacity models
│   │   └── external_devices/       # YOLO device models
│   ├── Patient_images/             # Uploaded X-ray images
│   ├── classification/             # Classification results
│   ├── segmentation/               # Segmentation outputs
│   ├── heatmap/                    # Heatmap visualizations
│   ├── opacity/                    # Opacity detection results
│   └── external_devices/           # Device detection results
│
├── yolo_models/                    # YOLOv5 model architecture
│   ├── common.py                   # Common YOLO layers
│   ├── experimental.py             # Experimental features
│   ├── yolo.py                     # YOLO model definitions
│   └── *.yaml                      # Model configurations
│
├── utils/                          # Utility functions
│   ├── general.py                  # General utilities
│   ├── torch_utils.py              # PyTorch utilities
│   ├── dataloaders.py              # Data loading
│   └── ...
│
├── config.py                       # Application configuration
├── main.py                         # Main Flask application
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended for faster inference)
- 8GB+ RAM
- 10GB+ free disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/arssaif/ai_grader.git
cd ai_grader
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Initialize Database

```bash
python -c "from config import app, db; app.app_context().push(); db.create_all()"
```

### Step 5: Run the Application

```bash
python main.py
```

The application will be available at `http://localhost:5000`

## 💻 Usage

### For Doctors

1. **Register/Login**: Create an account or log in with existing credentials
2. **Upload X-Ray**: Upload patient chest X-ray images (JPEG/PNG format)
3. **Run Analysis**: Click on the patient name and select analysis options:
   - Quick Analysis: Disease classification + Caption
   - Full Report: All diagnostic modules
4. **View Results**: Review the comprehensive diagnostic report with:
   - Disease probabilities
   - Automated caption
   - Segmentation with CTR
   - Heatmap visualization
   - Opacity detection
   - Device detection

### For Administrators

1. **User Management**: View and manage registered doctors
2. **System Monitoring**: Monitor application usage and performance
3. **User Removal**: Remove inactive or unauthorized users

### API Usage

#### Register User
```bash
POST /api/signup
Content-Type: application/x-www-form-urlencoded

fname=John&lname=Doe&email=john@example.com&password=securepass
```

#### Login
```bash
POST /api/login
Content-Type: application/x-www-form-urlencoded

email=john@example.com&password=securepass
```

#### Get Analysis
```bash
GET /getdat?p_name=patient_image.jpg
Authentication: Required (Session-based)
```

## 🤖 AI Models

### 1. Disease Classification
- **Architecture**: EfficientNetB4
- **Input**: 380x380 RGB images
- **Output**: Probabilities for 12 disease classes
- **Training Dataset**: ChestX-ray14

### 2. Image Captioning
- **Architecture**: CNN Encoder + GRU Decoder with Global Attention
- **Encoder**: CheXNet (DenseNet121)
- **Decoder**: GRU with attention mechanism
- **Output**: Natural language descriptions

### 3. Segmentation
- **Architecture**: U-Net
- **Models**: 3 separate models (Lungs, Heart, Clavicles)
- **Input**: 512x512 grayscale images
- **Loss Function**: Dice coefficient loss
- **Post-processing**: CTR calculation

### 4. Heatmap Generation
- **Method**: Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Base Model**: VGG16
- **Purpose**: Visual explanation of predictions

### 5. Opacity Detection
- **Architecture**: YOLOv5
- **Purpose**: Detect and localize lung opacities
- **Output**: Bounding boxes with confidence scores

### 6. External Device Detection
- **Architecture**: YOLOv5
- **Purpose**: Detect cardiac devices and implants
- **Classes**: Pacemakers, tubes, and other medical devices

## 📚 API Documentation

### Authentication Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/login` | Login page | No |
| GET | `/signup` | Registration page | No |
| POST | `/api/login` | Login API | No |
| POST | `/api/signup` | Registration API | No |
| GET | `/logout` | Logout user | Yes |

### Application Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | Home dashboard | Yes |
| GET | `/home` | Home dashboard | Yes |
| GET | `/AdminHome` | Admin panel | Yes (Admin) |
| POST | `/upload` | Upload patient image | Yes |
| POST | `/patient_name` | Get patient ID | Yes |
| GET | `/getdat` | Get analysis data | Yes |
| GET | `/getsegment` | Perform segmentation | Yes |
| GET | `/opacity` | Analyze opacity | Yes |
| GET | `/external_devices` | Detect devices | Yes |
| GET | `/get_full_report` | Generate full report | Yes |
| GET | `/get_email` | Delete user | Yes (Admin) |

## 🛠️ Technologies Used

### Backend
- **Flask**: Web framework
- **Flask-Login**: User authentication
- **Flask-RESTful**: REST API development
- **SQLAlchemy**: ORM for database operations
- **SQLite**: Database

### AI/ML Frameworks
- **TensorFlow 2.15.0**: Deep learning framework
- **Keras**: High-level neural networks API
- **PyTorch 2.1.2**: Deep learning framework
- **Torchvision**: Computer vision utilities
- **EfficientNet**: Efficient neural network architecture

### Computer Vision
- **OpenCV**: Image processing
- **scikit-image**: Image processing algorithms
- **matplotlib**: Visualization

### Utilities
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **joblib**: Model serialization
- **PyYAML**: Configuration management

## 📸 Youtube Demo
[![Demo Video](https://img.youtube.com/vi/DOOb9337-p4/0.jpg)](https://www.youtube.com/watch?v=DOOb9337-p4&autoplay=1)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines for Python code
- Add docstrings to all functions and classes
- Write meaningful commit messages

## 👨‍💻 Authors

- **Arslan Saif** - *Initial work* - [Arslan Saif](https://github.com/arssaif)

## 🙏 Acknowledgments

- ChestX-ray14 dataset for training data
- YOLOv5 team for object detection framework
- TensorFlow and PyTorch communities
- Open-source contributors

## 📞 Contact

For questions or support, please contact:
- Email: your.email@example.com
- GitHub Issues: [Project Issues](https://github.com/yourusername/chest-xray-diagnostic-app/issues)

## 🔮 Future Enhancements

- [ ] Integration with DICOM viewers
- [ ] Multi-language support
- [ ] Mobile application
- [ ] Real-time collaboration features
- [ ] Integration with hospital information systems (HIS)
- [ ] Advanced analytics and reporting dashboard
- [ ] Support for other imaging modalities (CT, MRI)

---

**⚠️ Medical Disclaimer**: This application is intended for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.
