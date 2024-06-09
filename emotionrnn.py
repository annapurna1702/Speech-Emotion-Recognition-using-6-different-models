import torch
import torch.nn as nn
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

def extract_features(file_path, max_length=100):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

        features = np.vstack([mfccs, chroma, spectral_contrast])
        if features.shape[1] < max_length:
            features = np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode='constant')
        elif features.shape[1] > max_length:
            features = features[:, :max_length]
        return features.T  # Transpose to shape [max_length, num_features]
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

class EmotionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EmotionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(hidden_size*2, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)  # Bidirectional
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)  # Bidirectional

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Forward propagate GRU
        out, _ = self.gru(out)

        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.relu(out)
        out = self.fc(out)
        return out

# Load the pretrained model
model_path = 'C:/Users/green/Desktop/my code AI/RNN_exec_model.pth'
input_size = 32  # 13 MFCC + 12 chroma + 14 spectral contrast
hidden_size = 128
num_layers = 2
num_classes = len(['sad', 'notsad'])  # Define your emotion labels

emotion_model = EmotionRNN(input_size, hidden_size, num_layers, num_classes)
emotion_model.load_state_dict(torch.load(model_path))
emotion_model.eval()

# Initialize the label encoder
emotions = ['notsad', 'sad']
label_encoder = LabelEncoder()
label_encoder.fit(emotions)

# Load a random .wav audio file and extract features
audio_file_path = 'videoplayback Ringtone [vocals].wav'  # Replace with the path to your audio file
features = extract_features(audio_file_path, max_length=100)

if features is not None:
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        logits = emotion_model(features)
        predicted_label_idx = torch.argmax(logits, dim=1).cpu().numpy()[0]
if predicted_label_idx==0:

    print(f'Predicted Emotion: sad')
elif predicted_label_idx==1:
    print(f"Predicted emotion:Not sad")
else:
    print('Failed to extract features from the audio file.')
