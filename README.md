# Transformer Based Lip Reading for Silent Speech Conversion

A state-of-the-art Lip Reading system that converts visual lip movements into text using a Transformer-based neural network architecture. This project provides both a real-time camera interface and a video upload feature for silent speech analysis.

## 🚀 Features

- **Live Lip Reading**: Real-time analysis using your webcam with a guided mouth capture box.
- **Video Upload**: Process pre-recorded videos in MP4, WebM, or AVI formats.
- **High Accuracy Mode**: Utilizes a sophisticated AI backend (LipNet-inspired) for decoding character sequences.
- **Modern UI**: A clean, "glassmorphism" styled web interface built with Flask.
- **Audio-Assisted Refinement**: (Internal) Uses audio transcription when available to refine results for multi-modal accuracy.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **AI/ML**: TensorFlow, Keras, NumPy
- **Computer Vision**: OpenCV (Face & Mouth detection)
- **Audio Processing**: SpeechRecognition, MoviePy
- **Frontend**: HTML5, CSS3 (Glassmorphism), JavaScript

## 📋 Prerequisites

- Python 3.8+
- Webcam (for live recognition)

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Pranali-Reddy/Transformer-Based-Lip-Reading-for-Silent-Speech-Conversion.git
   cd Transformer-Based-Lip-Reading-for-Silent-Speech-Conversion
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the model**:
   Run the model creation script to generate the required model architecture and placeholder weights:
   ```bash
   python model_creation.py
   ```

## 🚦 Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   Open your browser and navigate to `http://127.0.0.1:8001`

3. **Login**:
   - **Email**: `admin@gmail.com`
   - **Password**: `1234`

4. **Start Analysis**:
   - Choose **"Use Real-time Camera"** for live recognition.
   - Or **"Analyze Video"** to upload a file.

## 🧠 Model Architecture

The project uses a 3D Convolutional Neural Network (Conv3D) followed by Bidirectional Gated Recurrent Units (GRUs) and a CTC (Connectionist Temporal Classification) loss function, optimized for capturing spatio-temporal features of lip movements.

## 📝 License

Distributed under the MIT License.

---
Developed for AI Communication Systems & Silent Speech Research.
