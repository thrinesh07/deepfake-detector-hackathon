# DeepFake Detector – Hackathon Submission

This project is a **high-accuracy DeepFake Detection System** built from scratch using **TensorFlow and handcrafted features**, developed for the AI-based Threat Modeling / DeepFake Detection Hackathon.

---

## Overview

The model detects **real vs. fake (AI-generated)** images using:
- CNNs trained **from scratch** (no pretrained models)
- Fusion of **image features + handcrafted texture descriptors**
- **Feature scaling**, **stacking ensemble**, and **calibration** for robust predictions
- **Test-Time Augmentation (TTA)** for better generalization
- Final output as a JSON file matching the hidden ground-truth format

---

##  Features
 Trains 3 CNNs (A, B, C) from scratch  
 Uses handcrafted features (Laplacian, Sobel, Entropy)  
 Ensemble + Ridge stacking for improved closeness  
 Youden’s J adaptive threshold calibration  
 Produces final `ythrinesh_prediction.json` file  

---

## Folder Structure

DeepFake-Detector/
│
├── main.py                               
├── requirements.txt                      
├── README.md                            
├── .gitignore                            
│
├── data/                                 
│   ├── real_cifake_images/               
│   ├── fake_cifake_images/
│   └── test/
│__ results/
|    |__Model Summary.png
|    |__Training Logs.png
|    |__Validation Accuracy.png
|
├── outputs/                              
│   └── ythrinesh_prediction.json       
                            

