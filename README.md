# Emotion_detection
Emotion Detection from Facial Expressions using CNN
This project performs facial emotion recognition using the FER-2013 dataset and deep learning models including custom CNNs and transfer learning approaches.


Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

📌 Project Features
✅ Data preprocessing & cleaning

✅ Exploratory Data Analysis (EDA) & visualization

✅ Data augmentation using ImageDataGenerator

✅ Custom CNN built from scratch

✅ CNN with data augmentation

✅ Transfer Learning using VGG16 and ResNet50

✅ Evaluation using accuracy, loss, confusion matrix, and classification report

✅ Trained model saving and callback integration

🧠 Models Implemented
1. Custom CNN from Scratch
Deep CNN with batch normalization, dropout, and L2 regularization.

Trained on grayscale FER-2013 images.

2. Custom CNN with Augmentation
Incorporates rotation, shift, zoom, flip augmentations.

Improved generalization and performance on validation data.

3. VGG16 & ResNet50 Transfer Learning
Uses pretrained weights from ImageNet.

Fine-tuned on FER-2013 for emotion classification.

🧪 Evaluation Metrics
Accuracy and Loss curves

Confusion Matrix heatmap

Classification Report including precision, recall, and F1-score

Random Prediction Samples visualization

🧰 Technologies Used
Python

TensorFlow / Keras

OpenCV / PIL

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

🔧 How to Run
Clone the repository:
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
Download the FER-2013 dataset from Kaggle and extract it in the working directory.

Run the notebook or Python script:
python emotion_detection.py
📈 Results
Custom CNN with augmentation achieved up to ~70% accuracy.

Transfer Learning (VGG16/ResNet50) showed competitive or improved performance with less training time.

📸 Sample Output
True Label	Predicted	Image
Happy	Happy	✅
Sad	Angry	❌

Green = Correct Prediction, Red = Incorrect

