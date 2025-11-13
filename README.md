Deep Learning-Based Facial Emotion Recognition for Mental Health Monitoring 

## Overview:
This project aims to develop a Facial Emotion Recognition (FER) system using Deep Learning techniques to monitor emotional states as an indicator for mental health tracking.
The system detects facial expressions from images or live video feeds and classifies them into basic emotion categories such as Angry, Disgust, Fear, Happy, Neutral, Sad and Surprise.
By analyzing these emotions over time, the system can support mental health awareness and early detection of emotional distress.

## Dataset:
The project uses the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) which contains grayscale images of faces labeled with emotions.  
- **Training images**: 28,709  
- **Test images**: 7,178  
- Images are 48x48 pixels, grayscale.

## Installation & Setup
1. Clone the repository
git clone https://github.com/daanishh69/facial-emotion-recognition-fyp.git
cd facial-emotion-recognition-fyp

2. Set up environment - Using Anaconda:
conda create -n fer_new python=3.12
conda activate fer_new
pip install -r requirements.txt

3. Run training script (optional, if want to retrain the model)
python src/train_model.py

4. Test the model on an image
python src/test_model.py

## You can install all dependencies using:

pip install -r requirements.txt

## Weekly Progress Plan:

**Week 1-3** : Finalized FYP proposal document that has been reviewed and signed by my supervisor.

**Week 4**   : So far, Chapter 1 (Introduction) has been completed for Report Writing, including the background, problem statement, objectives, and scope. The development environment has been fully set up with Anaconda, VS Code, and Git. A dedicated environment named fer_env was created with all the necessary libraries, along with a requirements.txt file for easy setup. Version control was implemented through Git and connected to GitHub. A detailed README.md file was added, describing the project overview, objectives, methodology, tools, and weekly plan. The project folder was also organized into clear directories for data, models, notebooks, source code, and reports, and a .gitignore file was included to keep the repository tidy.

**Week 5**   : For this phase, I focused on completing the literature review and requirement analysis for the project. I reviewed several studies and existing systems related to facial emotion recognition and mental health monitoring to understand the methods, datasets, and models commonly used. The project methodology was outlined, covering data collection, preprocessing, model training, and evaluation processes. I also analyzed the system’s requirements by identifying the data, functional, and non-functional needs based on the project objectives. Tools, hardware specifications, and performance expectations were clearly defined to ensure the system can run efficiently. This stage helped establish a solid foundation for the upcoming system design and implementation phases.

**Week 6**   : This week was focused on the implementation and testing of the facial emotion recognition model. I successfully built and trained a convolutional neural network (CNN) using the FER-2013 dataset. The model architecture included multiple convolutional, pooling, batch normalization, dropout, and dense layers, optimized using the Adam optimizer with categorical cross-entropy loss. Training was executed for 50 epochs, achieving a working model saved as emotion_model.keras in the models directory.

I also created scripts for testing the model on individual images (test_model.py) and verifying the model architecture (verify_model.py). The project repository was updated with all new scripts, organized directories, and committed to GitHub with proper version control. The workflow included resolving environment issues, managing dependencies, and ensuring smooth data preprocessing with ImageDataGenerator.

This week’s tasks established the core system for facial emotion recognition, enabling both training and testing, and laid the groundwork for future evaluation, visualization, and potential model improvement in the upcoming weeks.
