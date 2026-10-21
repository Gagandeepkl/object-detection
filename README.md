# Real-Time Object Detection Using YOLO and Computer Vision

An enterprise-grade, beginner-friendly, real-time computer vision web application built with **Ultralytics YOLOv8**, **OpenCV**, and **Streamlit**. The system detects and localizes common objects in static photos and live video feeds with high accuracy and low latency.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Project Objectives](#project-objectives)
3. [Key Features](#key-features)
4. [Technology Stack](#technology-stack)
5. [System Requirements](#system-requirements)
6. [Project Folder Structure](#project-folder-structure)
7. [Installation & Setup](#installation--setup)
   - [Windows Setup (PowerShell)](#windows-setup-powershell)
   - [Linux / macOS Setup (Bash)](#linux--macos-setup-bash)
8. [Model Weight Download Behavior](#model-weight-download-behavior)
9. [Running the Application](#running-the-application)
10. [User Guide & Workflows](#user-guide--workflows)
    - [Image Detection](#image-detection)
    - [Live Webcam Detection](#live-webcam-detection)
    - [Adjusting Confidence Threshold](#adjusting-confidence-threshold)
    - [Saving & Downloading Annotated Outputs](#saving--downloading-annotated-outputs)
11. [Running Automated Tests](#running-automated-tests)
12. [Expected Output Examples](#expected-output-examples)
13. [Webcam Permissions & Cloud Limitations](#webcam-permissions--cloud-limitations)
14. [Troubleshooting Guide](#troubleshooting-guide)
15. [Limitations & Future Enhancements](#limitations--future-enhancements)


---

## Problem Statement

Traditional object recognition algorithms often require separate classification and localization passes, resulting in sluggish processing speeds unsuitable for real-time safety, surveillance, or robotics applications. Furthermore, machine learning pipelines often remain trapped inside technical scripts, lacking intuitive graphical user interfaces for non-technical stakeholders or students.

This project addresses these challenges by developing an accessible, modular, and reactive web application that integrates state-of-the-art single-stage deep learning (YOLOv8) with an interactive web dashboard for real-time inference on images and webcam streams.

---

## Project Objectives

- **High-Speed Object Detection**: Perform real-time object detection and localization at 30+ FPS on standard CPU hardware using pretrained YOLO weights.
- **Interactive Multi-Modal Input**: Support drag-and-drop image uploads (`.jpg`, `.jpeg`, `.png`) alongside live continuous webcam feeds.
- **Configurable Sensitivity**: Enable users to dynamically adjust the confidence threshold slider ($0.0$ to $1.0$) to fine-tune detection sensitivity.
- **Comprehensive Detection Telemetry**: Compute and present detailed telemetry including bounding box coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$, confidence scores, per-class counts, and processing FPS.
- **Robust Error Handling**: Handle missing hardware, corrupted files, and camera permission denials gracefully without crashing.
- **Architectural Rigor**: Maintain strict separation of concerns across UI, model inference, image processing, and webcam device management, backed by automated unit tests.

---

## Key Features

- **Pretrained YOLOv8 Engine**: Utilizes Ultralytics YOLOv8 Nano (`yolov8n.pt`) with full support for swapping to Small (`yolov8s.pt`) or Medium (`yolov8m.pt`) models.
- **Zero-Wait In-Memory Caching**: Implements Streamlit resource caching (`@st.cache_resource`) to load model weights into RAM once.
- **Interactive Image Analysis**:
  - Side-by-side comparison of original and annotated images.
  - One-click testing using bundled real benchmark images (`person.jpg`, `dog.jpg`, `bus.jpg`, `blank_texture.jpg`).
  - High-resolution visual annotations with colored bounding boxes and contrast labels.
  - Instant JPEG export of annotated images.
- **Live Webcam Processing**:
  - DirectShow / OpenCV frame acquisition with real-time HUD (Processing FPS, Objects in View, Active Classes).
  - Clean resource management with context handlers ensuring cameras are never left running.
  - Browser-native camera fallback for sandboxed cloud or Docker environments.
- **Automated Pytest Suite**: Covers core test cases TC1 through TC6 testing person detection, vehicle/animal detection, empty scene handling, strict thresholding, and camera failures.

---

## Technology Stack

| Layer | Component | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.10 / 3.11 | Primary language runtime |
| **Object Detection** | Ultralytics YOLOv8 | $\ge 8.0.0$ | State-of-the-art single-stage object detection model |
| **Deep Learning** | PyTorch & Torchvision | $\ge 2.0.0$ | Tensor computation backend |
| **Computer Vision** | OpenCV (`cv2`) | $\ge 4.8.0$ | Frame capture, bounding box rendering, color space conversion |
| **Image Processing** | Pillow (`PIL`) | $\ge 10.0.0$ | Image validation, byte decoding, and formatting |
| **Data Array Ops** | NumPy | $\ge 1.24.0$ | Numerical multi-dimensional matrix manipulation |
| **Web Dashboard** | Streamlit | $\ge 1.30.0$ | Reactive web application frontend |
| **Testing** | Pytest | $\ge 7.0.0$ | Automated unit and integration test framework |

---

## System Requirements

- **Operating System**: Windows 10/11, macOS 11+, or Linux (Ubuntu 20.04+)
- **CPU**: Dual-Core 2.0 GHz or higher (Intel Core i3/i5/i7, AMD Ryzen, or Apple Silicon)
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Disk Space**: ~250 MB free space (including model weights and virtual environment)
- **Camera (Optional)**: Integrated webcam or external USB camera for live video mode

---

## Project Folder Structure

```
realtime_object_detection/
├── app.py                     # Main Streamlit application and UI layout
├── config.py                  # Global configurations, model paths, and UI constants
├── requirements.txt           # Verified Python package dependencies
├── README.md                  # Comprehensive documentation and viva guide
├── .gitignore                 # Ignored files, cache, and virtual environments
├── yolov8n.pt                 # Pretrained YOLOv8 Nano weights file (~6.5MB)
│
├── src/                       # Core source code modules
│   ├── __init__.py            # Package interface and exports
│   ├── detector.py            # YOLODetector class, inference engine, result structures
│   ├── image_utils.py         # Validation, byte decoding, annotations, and export
│   └── webcam.py              # OpenCV VideoCapture lifecycle, error handling, FPS counter
│
├── tests/                     # Automated unit and integration test suite
│   ├── __init__.py            # Test package marker
│   ├── test_detector.py       # Automated tests TC1 - TC6 (Person, Car, Empty, 0.90 conf, etc.)
│   └── test_image_utils.py    # Unit tests for image validation, decoding, and drawing
│
├── sample_images/             # Bundled benchmark test images
│   ├── README.md              # Documentation of bundled images and expected outputs
│   ├── person.jpg             # Portrait benchmark image (detects 2 persons)
│   ├── dog.jpg                # Multi-object benchmark (dog, bicycle, car, truck)
│   ├── bus.jpg                # Urban street benchmark (bus, pedestrians, stop sign)
│   └── blank_texture.jpg      # True negative benchmark (0 detected objects)
│
└── outputs/                   # Directory where annotated images are saved
    └── .gitkeep               # Preserves output folder structure in git
```

---

## Installation & Setup

### Windows Setup (PowerShell)

1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd c:\Users\HP\Downloads\object
   ```

2. *(Optional but Recommended)* Create and activate a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install the required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

### Linux / macOS Setup (Bash)

1. Open a terminal and navigate to the project directory:
   ```bash
   cd /path/to/project
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Model Weight Download Behavior

- The project comes pre-configured to use `yolov8n.pt` located in the project root directory.
- If the weights file is absent, the underlying **Ultralytics** engine will automatically download the official pre-trained checkpoint from GitHub when an internet connection is detected.
- If offline and weights are missing or corrupt, `ModelLoadError` is raised with explicit user instructions rather than masking the failure.

---

## Running the Application

Execute the following command from the project root:

```bash
python -m streamlit run app.py
```

### Accessing the Local Webpage
Upon starting, Streamlit will print the local server URL. Open your preferred web browser and navigate to:
```
http://localhost:8501
```

---

## User Guide & Workflows

### Image Detection
1. In the sidebar, select **🖼️ Image Detection**.
2. **Upload an Image**: Drag and drop a `.jpg`, `.jpeg`, or `.png` file into the upload zone, **OR**
3. **Use Included Sample Images**: Switch to the **Test Included Sample Images** tab and choose `person.jpg`, `dog.jpg`, or `bus.jpg`.
4. The application executes inference, renders side-by-side original and annotated displays, shows total detected counts, lists identified categories with badges, and populates a granular bounding box data table.

### Live Webcam Detection
1. In the sidebar, select **📹 Live Webcam**.
2. Choose **Continuous Live Stream (OpenCV)** for real-time video.
3. Select your camera index (typically `0` for built-in webcams).
4. Click **▶️ Start Webcam Detection**.
5. The video feed streams annotated frames with real-time FPS and object count updates.
6. Uncheck **Start Webcam Detection** to halt the feed and immediately release the camera device.
7. *Alternative*: Select **Webcam Snapshot (Browser Camera)** if running in a browser sandbox or remote server.

### Adjusting Confidence Threshold
- Use the **Confidence Threshold** slider in the sidebar (range: `0.0` to `1.0`, step `0.05`).
- **Initial Default**: `0.25` (Standard balanced sensitivity).
- **High Threshold (0.90)**: Eliminates false positives, displaying only detections with $\ge 90\%$ certainty. Note that partially obscured or smaller objects may be omitted.

### Saving & Downloading Annotated Outputs
- On the Image Detection page, click the **📥 Download Annotated Image** button below the annotated output to save the annotated JPEG directly to your computer.
- The `src.image_utils.save_annotated_image` helper automatically saves timestamped outputs to the `outputs/` folder.

---

## Running Automated Tests

The test suite validates compliance with requirements TC1 through TC6 and all image utility functions.

Execute pytest from the root folder:

```bash
python -m pytest tests/ -v
```

### Test Case Verification Summary
| Test ID | Test Target | Description | Status |
| :--- | :--- | :--- | :--- |
| **TC1** | `test_tc1_person_detection` | Validates detection of `person` on `person.jpg` | ✅ PASSED |
| **TC2** | `test_tc2_car_or_dog_detection` | Validates detection of `dog` / `car` on `dog.jpg` | ✅ PASSED |
| **TC3** | `test_tc3_unrecognizable_objects_empty_detections` | Validates `0 detections` on `blank_texture.jpg` | ✅ PASSED |
| **TC4** | `test_tc4_high_confidence_threshold_filtering` | Verifies strict 0.90 threshold filters lower scores | ✅ PASSED |
| **TC5** | `test_tc5_nonexistent_image_path_error` | Verifies controlled `FileNotFoundError` on missing path | ✅ PASSED |
| **TC6** | `test_tc6_unavailable_webcam_error_handling` | Verifies controlled `WebcamAccessError` on missing camera | ✅ PASSED |

---

## Expected Output Examples

### Image Detection Example (`dog.jpg` at 0.25 Confidence)
- **Total Objects**: 4
- **Detected Classes**: `dog` (1), `bicycle` (1), `car` (1), `truck` (1)
- **Confidence Scores**:
  - `dog`: 90.8%
  - `bicycle`: 88.9%
  - `car`: 53.1%
  - `truck`: 50.9%
- **Inference Time**: ~40-60 ms (Intel CPU)

### Image Detection Example (`person.jpg` at 0.25 Confidence)
- **Total Objects**: 3
- **Detected Classes**: `person` (2), `tie` (1)
- **Confidence Scores**: `person` (83.6%), `person` (81.9%), `tie` (29.1%)

### True Negative Example (`blank_texture.jpg` at 0.25 Confidence)
- **Total Objects**: 0
- **UI Message**: *"No qualifying objects detected. Try lowering the confidence slider..."*

---

## Webcam Permissions & Cloud Limitations

- **Local Windows / macOS / Linux**: OpenCV requires OS-level permission to access the camera hardware. In Windows 10/11, ensure **Settings > Privacy & Security > Camera > Let desktop apps access your camera** is enabled.
- **Concurrent Access**: Webcams cannot be shared between multiple applications. Close Zoom, Microsoft Teams, or Skype before starting webcam detection.
- **Cloud / Docker Deployments**: Remote environments (e.g., Streamlit Community Cloud, AWS EC2, Docker) do not have direct access to a client user's physical camera hardware. For cloud deployments, use the included **Webcam Snapshot (Browser Camera)** mode, which streams frames via the client's web browser.

---

## Troubleshooting Guide

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `WebcamAccessError` or Black Screen | Camera in use or permission denied | Close background apps (Teams, Zoom). Verify Windows Camera privacy settings. |
| `InvalidImageError: Unsupported file format` | User uploaded non-image file (e.g. PDF, GIF) | Upload standard `.jpg`, `.jpeg`, or `.png` files. |
| `ModelLoadError` | Missing or corrupted `.pt` file | Ensure `yolov8n.pt` is in the project root or verify internet connection for automatic download. |
| Low Processing FPS on Webcam | High camera resolution or CPU throttling | Resolution is standardized to $640 \times 480$ for efficiency. Ensure battery settings are set to High Performance. |
| Port 8501 Already in Use | Another Streamlit instance is running | Run on an alternate port: `python -m streamlit run app.py --server.port 8502` |

---

## Limitations & Future Enhancements

### Current Limitations
- **COCO Class Boundary**: Pretrained weights detect the 80 standard COCO classes. Niche industrial or medical items require custom transfer learning.
- **Lighting Sensitivity**: Low-light camera feeds or heavy motion blur reduce detection confidence.

### Future Enhancements
- **Custom Model Fine-Tuning**: Integration of a training tab to fine-tune weights on custom user datasets.
- **Object Tracking (DeepSORT / ByteTrack)**: Persistent object ID tracking across consecutive video frames to count unique visitors or vehicles.
- **Zone Intrusion & Polygon Alerts**: Triggering visual/audio alerts when objects enter user-defined virtual boundaries.

---

