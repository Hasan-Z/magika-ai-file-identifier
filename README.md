# 🛡️ Magika AI File Identifier

A high-performance File Identification Web API and Interface leveraging [Google's Magika](https://github.com/google/magika) deep learning model for fast, accurate format detection.

---

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-v1.0.0-005571?style=flat-square&logo=fastapi)
![Magika](https://img.shields.io/badge/Magika-v1.0+-4285F4?style=flat-square&logo=google)
![License](https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square)

---

## 🏗️ Engineering Architecture: Hybrid Slicing

Unlike traditional tools that read entire files, this app utilizes **Hybrid Byte Slicing** via the [Magika Python API](https://google.github.io/magika/).

* **Network Efficiency:** Extracts and transmits optimized byte slices (**512B Head**, **512B Mid**, **512B Tail**) directly from client side.
* **Recursive Folder Inspection:** Process nested directory trees natively with relative path preservation.
* **Constant-Time O(1) Processing:** Analysis latency remains virtually fixed whether processing a 10KB script or a 50GB disc image.

---

## 🚀 Quick Start

### 1. Requirements
* **Python 3.10+**
* Google Chrome, Edge, or Firefox

### 2. Launching the App

#### **Windows**
Double-click `run.bat` or execute in terminal:
```cmd
run.bat
```

#### **macOS / Linux**
1. Open your terminal in the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   python main.py
   ```
4. Open your browser to: `http://127.0.0.1:8000`

---

## ✨ Key Features
* **Recursive Folder Drop**: Drag and drop whole directory hierarchies while keeping nested file paths intact.
* **Persistent Theme Toggle**: Seamless **Dark/Light** mode switcher saved in `localStorage`.
* **Forensic Metadata Inspection**: Clean display of MIME types, detection confidence scores, and expandable raw JSON model outputs.
* **Extension Mismatch Detector**: Dedicated endpoint (`/detect_extension_mismatch`) to catch spoofed or modified extensions.
* **About & Project Modal**: Quick overlay panel displaying app version (`v1.0.2`) and developer info.

---

## 🧪 Testing
Run unit tests on **Windows**:
```cmd
run-tests.bat
```

Or execute via **PyTest** (macOS / Linux / Windows):
```bash
pytest test_main.py -v
```

---

## 📝 Technical Specifications & Author
* **AI Engine**: [Google Magika](https://opensource.google/projects/magika) (v1.0+ ONNX execution)
* **Backend Framework**: FastAPI & Uvicorn
* **Author**: Hasan Zemzem | [hasan.zamzam@gmail.com](mailto:hasan.zamzam@gmail.com)

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.