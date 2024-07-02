import numpy as np
import librosa
import torch
import joblib  # Import joblib to load the scikit-learn model
from sklearn.metrics import classification_report
import os

# Load the decision tree model saved as a .pt file
model_path = "C:/Users/green/Desktop/my code AI/DTexecmodel.pt"
decision_tree_model = decision_tree_model = torch.load(model_path)  # Load thejoblib.load(model_path) scikit-learn model using joblib

# Define emotions
emotions = [ 'sad', 'notsad']

# Function to extract features from audio file (you can reuse your existing function)
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        return mfccs_mean
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

# Load data and extract features (you can reuse your existing code)
X, y = [], []
for file in os.listdir("audiosets"):
    if file.endswith(".wav"):
        emotion = file.split("_")[0]  # Extract emotion label from file name
        features = extract_features(os.path.join("audiosets", file))
        if features is not None:
            X.append(features)
            y.append(emotion)
X = np.array(X)
y = np.array(y)



# Define the path to the audio file you want to test
test_audio_file = "1001_DFA_SAD_XX.wav"

# Extract features from the test audio file
test_features = extract_features(test_audio_file)

# Perform inference using the loaded decision tree model
predicted_class_index = decision_tree_model.predict([test_features])[0]
# Convert predicted class index to integer
if predicted_class_index=='sad':
   predicted_class_index = 0
else:
    predicted_class_index=1



# Convert predicted class index to emotion label
if predicted_class_index==0:
    predicted_emotion = 'sad'

else:
    predicted_emotion='not sad'

print(f"Predicted emotion for {test_audio_file}: {predicted_emotion}")


# Convert predicted class index to emotion label

