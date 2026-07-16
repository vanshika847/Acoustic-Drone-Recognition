# 🛸 Acoustic Drone Recognition and Threat Assessment System

> **An AI-powered acoustic intelligence system for real-time drone detection, classification, localization, and threat assessment using deep learning.**

---

## 📖 Overview

The rapid growth of Unmanned Aerial Vehicles (UAVs) has created new challenges in surveillance, security, and airspace monitoring. Conventional detection technologies such as radar, cameras, and RF-based systems often struggle with small, low-flying, or autonomous drones.

This project explores **acoustic sensing** as an alternative approach by analyzing the unique sound signatures produced by drone motors and propellers. By combining digital signal processing with deep learning, the system transforms environmental audio into actionable intelligence.

---

## 🎯 Key Features

* 🎧 Real-time drone detection
* 🚁 Drone category classification
* 📏 Drone size estimation
* ✈️ Flight movement recognition
* 🧭 Direction of arrival estimation
* 📍 Distance estimation
* ⚡ Speed estimation
* 👥 Multi-drone detection
* 🚨 Threat assessment engine
* 📊 Interactive monitoring dashboard
* 📈 Live spectrogram visualization
* 📜 Detection history and analytics

---

## 🏗 System Architecture

```text
                  Microphone Array
                         │
                         ▼
                Audio Acquisition
                         │
                         ▼
          Noise Reduction & Filtering
                         │
                         ▼
              Feature Extraction Layer
       (MFCC • Mel • Chroma • Spectral)
                         │
                         ▼
         Multi-Task Deep Learning Network
                         │
      ┌──────────┬─────────────┬──────────┐
      ▼          ▼             ▼          ▼
 Detection   Classification  Localization Threat Assessment
                         │
                         ▼
           Real-Time Dashboard & Alerts
```

---

## 🧠 AI Capabilities

The system is designed as a **multi-task learning framework** capable of simultaneously predicting:

* Drone Presence
* Drone Category
* Estimated Size
* Flight Movement
* Direction of Arrival
* Estimated Distance
* Estimated Speed
* Drone Count
* Threat Score
* Threat Level

---

## 🎵 Audio Features

The feature extraction pipeline includes:

* Mel Frequency Cepstral Coefficients (MFCC)
* Mel Spectrogram
* Chroma Features
* Spectral Centroid
* Spectral Contrast
* Spectral Roll-off
* RMS Energy
* Zero Crossing Rate (ZCR)
* Harmonic Features
* Tonnetz Features

---

## 📂 Project Structure

```text
Acoustic-Drone-Recognition/

├── api/
├── configs/
├── dashboard/
├── datasets/
├── docs/
├── experiments/
├── feature_extraction/
├── features/
├── logs/
├── models/
├── notebooks/
├── outputs/
├── preprocessing/
├── tests/
└── utils/
```

---

## 📊 Datasets

The project integrates multiple publicly available datasets to improve robustness and generalization.

* AeroSonicDB
* Kaggle Drone Audio
* ESC-50
* UrbanSound8K
* AudioSet
* IEEE SP Cup 2019 (DREGON)
* UaVirBASE
* DDL
* Freesound Drone Recordings

---

## 💻 Technology Stack

### Programming

* Python 3.11

### Deep Learning

* PyTorch
* TorchAudio

### Audio Processing

* Librosa
* NumPy
* SciPy
* SoundFile
* Noisereduce

### Machine Learning

* Scikit-learn

### Backend

* FastAPI

### Dashboard

* Streamlit

### Visualization

* Matplotlib
* Plotly

---

## 🎯 Use Cases

* 🛡 Border and perimeter surveillance
* 🏭 Industrial facility monitoring
* 🪖 Military and defense applications
* 🏢 Critical infrastructure protection
* 🛫 Airport and airspace monitoring
* 🎪 Public event security
* 🌳 Wildlife conservation and anti-poaching
* 🚨 Smart city surveillance
* 🔬 Academic research in acoustic AI
* 🤖 Edge AI deployment on embedded devices

---

## 🔬 Future Scope

* Transformer-based audio models
* Audio foundation models
* Drone model identification
* Drone swarm behavior analysis
* Sensor fusion (Radar + RF + Thermal + Acoustic)
* Edge deployment on NVIDIA Jetson and Raspberry Pi
* Cloud-based monitoring platform
* Mobile monitoring application
* Automatic emergency alert system

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome. Please open an issue or submit a pull request for discussions and enhancements.

---

## 📜 License

This project is intended for educational and research purposes. Individual datasets remain subject to their respective licenses.

---

## 👩‍💻 Author

**Vanshika Shinde**
**Powered by git commit -m "it finally works"**
Artificial Intelligence & Machine Learning Student


