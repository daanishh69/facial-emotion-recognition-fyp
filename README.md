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
**Week 1-3** : Finalized FYP proposal document that has been reviewed and signed by my supervisor.

**Week 4**   : So far, Chapter 1 (Introduction) has been completed for Report Writing, including the background, problem statement, objectives, and scope. The development environment has been fully set up with Anaconda, VS Code, and Git. A dedicated environment named fer_env was created with all the necessary libraries, along with a requirements.txt file for easy setup. Version control was implemented through Git and connected to GitHub. A detailed README.md file was added, describing the project overview, objectives, methodology, tools, and weekly plan. The project folder was also organized into clear directories for data, models, notebooks, source code, and reports, and a .gitignore file was included to keep the repository tidy.

**Week 5**   : For this phase, I focused on completing the literature review and requirement analysis for the project. I reviewed several studies and existing systems related to facial emotion recognition and mental health monitoring to understand the methods, datasets, and models commonly used. The project methodology was outlined, covering data collection, preprocessing, model training, and evaluation processes. I also analyzed the system’s requirements by identifying the data, functional, and non-functional needs based on the project objectives. Tools, hardware specifications, and performance expectations were clearly defined to ensure the system can run efficiently. This stage helped establish a solid foundation for the upcoming system design and implementation phases.

Repository Structure:

FYPPROJECT/

data/             # Dataset folder (FER2013, CK+, etc.)
notebooks/        # Jupyter notebooks for training and testing
models/           # Saved trained models (.h5 files)
src/              # Python scripts (preprocessing, training, dashboard)
reports/          # Report documents (Chapter 1–7)
README.md         # Project overview (this file)
requirements.txt  # Library dependencies
