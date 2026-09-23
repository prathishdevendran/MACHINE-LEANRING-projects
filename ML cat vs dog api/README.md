# Cat vs Dog Image Classification Web API & Deployment

A production-ready Deep Learning computer vision web service built with **FastAPI**, **TensorFlow / tf-keras**, and **Jinja2 Templates** to classify images of cats and dogs in real time with confidence scoring.

---

## 1. Project Overview & Architecture
This project delivers a complete machine learning deployment application that bridges trained Deep Learning models with a user-friendly web interface and RESTful API endpoints. 

### Key Capabilities
- **Deep Learning Model**: Pretrained convolutional neural network (`dogs_vs_cats.h5`) accepting 160x160 RGB image inputs.
- **Robust Model Serving**: Modular prediction pipeline in `prediction_model.py` with custom layer handling (`CustomDepthwiseConv2D`) ensuring cross-version compatibility across TensorFlow and `tf-keras`.
- **FastAPI Web Framework**: Asynchronous backend handling image uploads, dynamic template rendering, and REST responses.
- **Interactive User Interface**: Clean HTML/CSS frontend with live image preview (Base64 encoding) and animated classification badges displaying the predicted class and confidence percentage.

---

## 2. Directory Structure
```text
ML cat vs dog api/
├── dogs and cats image/      # Sample images for testing the service
├── templates/
│   └── index.html            # Jinja2 web interface template
├── dogs_vs_cats.h5           # Serialized Keras H5 model weights
├── main.py                   # FastAPI routing, file upload, and rendering logic
├── prediction_model.py       # Image preprocessing and inference engine
├── requirements.txt          # Python runtime dependencies
└── README.md                 # Deployment & setup documentation
```

---

## 3. Local Deployment & Execution Guide

Follow these step-by-step instructions to set up and run the service locally on your machine.

### Step 1: Navigate to the Project Directory
Open your terminal or command prompt and change directory into this folder:
```bash
cd "ML cat vs dog api"
```

### Step 2: Create a Python Virtual Environment
It is recommended to use Python 3.9, 3.10, or 3.11. Create an isolated virtual environment:
```bash
# On Windows / macOS / Linux
python -m venv venv
```

### Step 3: Activate the Virtual Environment
- **Windows (Command Prompt / PowerShell)**:
  ```powershell
  # PowerShell:
  .\venv\Scripts\Activate.ps1

  # CMD:
  .\venv\Scripts\activate.bat
  ```
- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

### Step 4: Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Launch the FastAPI Application
Start the Uvicorn ASGI server with automatic reload enabled:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Or directly via Python module:
```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

---

## 4. Accessing the Application & Endpoints

Once the server is running, open your web browser:

| Interface | URL | Description |
| :--- | :--- | :--- |
| **Web GUI** | `http://127.0.0.1:8000/` | Interactive image upload and prediction dashboard |
| **Interactive API Docs** | `http://127.0.0.1:8000/docs` | Swagger UI for testing endpoints interactively |
| **Alternative API Docs** | `http://127.0.0.1:8000/redoc` | ReDoc API specifications |

---

## 5. Inference Workflow & API Usage

1. Open `http://127.0.0.1:8000/` in your browser.
2. Click **Browse / Choose File** and select any sample image from the `dogs and cats image/` directory (or any `.jpg`/`.png` image of a cat or dog).
3. Click **Upload & Classify**.
4. The system processes the image in memory via PIL and NumPy, generates the model prediction, and displays the input image alongside the predicted class label (**Cat** or **Dog**) and confidence score.
