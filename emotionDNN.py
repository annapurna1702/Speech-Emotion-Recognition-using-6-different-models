import os
import numpy as np
import librosa
import torch
import torch.nn as nn

# Define emotions
emotions = ['sad', 'notsad']
max_length = 100

# Step 1: Extract Features from the Audio File
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

# Step 2: Load the Pretrained Model
class EmotionClassifierModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmotionClassifierModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Instantiate your DNN model
input_size = 13 * max_length  # 13 MFCC features times the maximum length
hidden_size = 128
num_classes = 2  # Number of emotions
model = EmotionClassifierModel(input_size, hidden_size, num_classes)

# Load the pretrained PyTorch model weights
model_path = 'C:/Users/green/Desktop/my code AI/DNN11model.pth'
model.load_state_dict(torch.load(model_path))

# Step 3: Run Inference
# Load a random .wav audio file and extract features
audio_file_path = 'sad11.wav'  # Replace with the path to your audio file
features = extract_features(audio_file_path, max_length=100)
predicted_emotion=''
if features is not None:
    # Reshape features to match the expected input size
    features = features.flatten()
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        logits = model(features.to('cpu'))  # Specify device here
        predicted_label_idx = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    if predicted_label_idx==0:
        predicted_emotion='not sad'
    if predicted_label_idx==1:
        predicted_emotion='sad'
    
    #print(predicted_label_idx)
    print(f'Predicted Emotion: {predicted_emotion}')
else:
    print('Failed to extract features from the audio file.')
