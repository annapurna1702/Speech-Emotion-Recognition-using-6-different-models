import os
import numpy as np
import librosa
import torch
import torch.nn as nn

# Step 1: Load the saved PyTorch model
model_path = 'C:/Users/green/Desktop/my code AI/MLP_exec_1model.pth'  # Path to your saved model
state_dict = torch.load(model_path)

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Step 2: Instantiate your MLP model
input_size = 13 * 100  # Assuming the same input size as your previous model
hidden_size = 256
num_classes = 2 # Number of emotions
model = MLPModel(input_size, hidden_size, num_classes)

# Step 3: Load the state_dict into your model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Step 4: Define a function to extract features from an audio file
def extract_features(file_path, max_length=100):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        elif mfccs.shape[1] > max_length:
            mfccs = mfccs[:, :max_length]
        return mfccs.T  # Transpose to shape [max_length, 13]
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

# Step 5: Load a random .wav audio file and extract features
audio_file_path = 'videoplayback Ringtone [vocals].wav'  # Replace with the path to your audio file
features = extract_features(audio_file_path)

# Step 6: Perform inference
if features is not None:
    # Reshape features to match the input size of the model
    features = features.flatten()
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        logits = model(features_tensor)
        predicted_label_idx = torch.argmax(logits, dim=-1).cpu().numpy()[0]

if predicted_label_idx==0:

    print(f'Predicted Emotion: sad')
elif predicted_label_idx==1:
    print(f"Predicted emotion:Not sad")
else:
    print('Failed to extract features from the audio file.')
