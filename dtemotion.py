import numpy as np
import librosa
import torch
from sklearn.preprocessing import StandardScaler

# Load the saved model
model_path = "DT_modelfinal.pt"
clf = torch.load(model_path)

# Define emotions
emotions = ['sad', 'notsad']
scaler = StandardScaler()

# Function to extract features from audio file
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        
        # Aggregate features: mean and variance
        features = np.hstack([
            np.mean(mfccs, axis=1), np.var(mfccs, axis=1),
            np.mean(chroma, axis=1), np.var(chroma, axis=1),
            np.mean(mel, axis=1), np.var(mel, axis=1),
            np.mean(contrast, axis=1), np.var(contrast, axis=1),
            np.mean(tonnetz, axis=1), np.var(tonnetz, axis=1)
        ])
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

# Example usage
test_audio_file = "mhappy.wav"
features = extract_features(test_audio_file)
if features is not None:
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    features = scaler.fit_transform(features)  # Standardize the features
    predicted_class = clf.predict(features)[0]
    print(f"Predicted class: {predicted_class}")
else:
    print("Failed to extract features from the audio file.")
