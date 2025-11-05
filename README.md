Deep Learning-Based Facial Emotion Recognition for Mental Health Monitoring

Overview:
This project aims to develop a Facial Emotion Recognition (FER) system using Deep Learning techniques to monitor emotional states as an indicator for mental health tracking.
The system detects facial expressions from images or live video feeds and classifies them into basic emotion categories such as Happy, Sad, Angry, Fear, Surprise, and Neutral.
By analyzing these emotions over time, the system can support mental health awareness and early detection of emotional distress.

Objectives: 
1. To design and develop a facial emotion recognition system using deep learning.
2. To train and evaluate a CNN model for emotion classification.
3. To integrate the model into a real-time monitoring interface for emotion tracking.

Methodology Overview: 
1. Data Collection – Using public datasets such as FER2013 and CK+.
2. Data Preprocessing – Face detection, resizing, and augmentation using OpenCV and Keras.
3. Model Development – CNN or Transfer Learning models (e.g., MobileNetV2, EfficientNetB0).
4. System Integration – Real-time emotion prediction via webcam using Python & OpenCV.
5. Dashboard Development – Visualization of emotion history for potential mental health insights.

Tools & Technologies: 
| Category             | Tools / Libraries   |
| -------------------- | ------------------- |
| Programming Language | Python 3.10         |
| Deep Learning        | TensorFlow, Keras   |
| Computer Vision      | OpenCV              |
| Data Handling        | NumPy, Pandas       |
| Visualization        | Matplotlib, Seaborn |
| Deployment           | Streamlit / Flask   |
| Version Control      | Git & GitHub        |
| Environment          | Anaconda, VS Code   |

Weekly Progress Plan: 
| Week        | Task                           | Description                                                            |
| ----------- | ------------------------------ | ---------------------------------------------------------------------- |
| **Week 4**  | Setup                          | Finalized title, set up GitHub, VS Code, and environment.              |
| **Week 5**  | Literature Review & Dataset    | Research FER & mental health papers, download datasets (FER2013, CK+). |
| **Week 6**  | Preprocessing & Baseline Model | Implement face detection, data augmentation, build simple CNN.         |
| **Week 7**  | Model Optimization             | Try transfer learning, tune parameters, evaluate accuracy.             |
| **Week 8**  | Real-Time Implementation       | Build webcam-based live emotion recognition demo.                      |
| **Week 9**  | Dashboard Development          | Integrate with Streamlit/Flask dashboard for visualization.            |
| **Week 10** | Testing & Evaluation           | Conduct tests with new data, analyze confusion matrix, model accuracy. |
| **Week 11** | Documentation                  | Write Chapter 2–6, finalize report content.                            |
| **Week 12** | Presentation & Submission      | Prepare final report, presentation slides, and upload deliverables.    |

Repository Structure:
FYPPROJECT/

data/             # Dataset folder (FER2013, CK+, etc.)
notebooks/        # Jupyter notebooks for training and testing
models/           # Saved trained models (.h5 files)
src/              # Python scripts (preprocessing, training, dashboard)
reports/          # Report documents (Chapter 1–7)
README.md         # Project overview (this file)
requirements.txt  # Library dependencies
