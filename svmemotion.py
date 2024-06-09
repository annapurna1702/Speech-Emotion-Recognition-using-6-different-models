import numpy as np
import librosa
import joblib

# Load the pre-trained SVM model and scaler
model_path = "SVMexec_modeltesting.pkl"
svm_model = joblib.load(model_path)
scaler = joblib.load('scaler.pkl')

# Define a function to extract features from an audio file
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

        # Aggregate features: mean and variance
        features = np.hstack([
            np.mean(mfccs, axis=1), np.var(mfccs, axis=1),
            np.mean(chroma, axis=1), np.var(chroma, axis=1),
            np.mean(spectral_contrast, axis=1), np.var(spectral_contrast, axis=1)
        ])
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

# Example usage:
test_audio_file = "audio comedy .wav"  # Replace with your test audio file
features = extract_features(test_audio_file)
if features is not None:
    # Normalize the features
    features = scaler.transform([features])

    # Perform inference using the SVM model
    predicted_class = svm_model.predict(features)[0]
    
    print(f"Predicted class: {predicted_class}")
else:
    print("Failed to extract features from the audio file.")
