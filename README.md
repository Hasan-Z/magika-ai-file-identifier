# 🛡️ Magika AI Identifier

A high-performance File Identification Web API and Interface leveraging [Google's Magika](https://github.com/google/magika) deep learning model.

---

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/python-3.8+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.110.0-005571?style=flat-square&logo=fastapi)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-v3.4-38B2AC?style=flat-square&logo=tailwind-css)
![License](https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square)

---

## 🏗️ Engineering Architecture: Hybrid Slicing

Unlike traditional tools that read entire files, this app uses **Hybrid Slicing** via the [Magika Python API](https://google.github.io/magika/).

* **Network Efficiency:** Transmits only **12KB** per file (4KB Head, 4KB Mid, 4KB Tail).
* **O(1) Complexity:** Identification speed is constant, whether the file is 1MB or 100GB.

---

## 🚀 Installation & Usage

### 1. Requirements
* **Python 3.8+** (Minimum)
* Google Chrome (recommended)

### 2. Launching the App

#### **Windows**
Double-click the `run.bat` file.

#### **macOS / Linux**
1. Open your terminal in the project folder.
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn magika python-multipart
   ```
3. Run the server:
   ```bash
   uvicorn main:app --reload
   ```
4. Open your browser to: `http://127.0.0.1:8000`

---

## ✨ Features
* **Apple-Inspired UI**: Glassmorphic interface with persistent **Light/Dark** modes.
* **Background Initialization**: The [Magika engine](https://pypi.org/project/magika/) warms up in a background thread to prevent UI blocking.
* **Developer Inspection**: Collapsible raw JSON response section for deep analysis.
* **Dynamic Icons**: Smart mapping to Lucide icons based on detected file types.

---

## 📝 Technical Specifications
* **AI Engine**: [Google Magika](https://opensource.google/projects/magika) (ONNX Runtime)
* **Frontend**: Tailwind CSS & Lucide Icons
* **License**: [Apache-2.0](https://github.com/google/magika/blob/main/LICENSE)