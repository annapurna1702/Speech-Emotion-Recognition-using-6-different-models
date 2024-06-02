from sklearn.svm import SVC
import numpy as np
import librosa
import torch

# Load the pre-trained SVM model
model_path = "C:/Users/green/Desktop/my code AI/SVM11model.pt"
svm_model = torch.load(model_path)


# Define a function to extract features from audio file
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        return mfccs_mean
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

# Example usage:
test_audio_file = "audio-4.wav"
features = extract_features(test_audio_file)
if features is not None:
    # Perform inference using the SVM model
    predicted_class = svm_model.predict([features])[0]
    
    print(f"Predicted class: {predicted_class}")
