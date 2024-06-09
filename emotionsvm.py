import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
from collections import Counter

# Define emotions
emotions = ['sad', 'notsad']

# Function to extract features from audio file
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

# Function to extract label from filename
def extract_label(filename):
    label = filename.split('_')[0]
    return label

# Initialize lists for features and labels
X, y = [], []

# Load data and extract features based on filenames
data_dir = 'audiosets'  # Replace with your data directory
for file in os.listdir(data_dir):
    if file.endswith(".wav"):
        features = extract_features(os.path.join(data_dir, file))
        if features is not None:
            label = extract_label(file)
            if label in emotions:
                X.append(features)
                y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Check class distribution
print(Counter(y))

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svc = SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)

print("Best parameters found:", clf.best_params_)

# Train SVM classifier with best parameters
cls = SVC(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'])
cls.fit(X_train, y_train)

# Predict on test set
y_pred = cls.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

# Generate classification report
report = classification_report(y_test, y_pred, target_names=emotions, zero_division=1)
print("Classification Report:")
print(report)

# Save the model
model_save_path = "SVMexec_model.pkl"
joblib.dump(cls, model_save_path)

print(f"Model saved as .pkl file at: {model_save_path}")
