import torch
import torch.nn as nn
import librosa
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


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
    


class EmotionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EmotionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Load the pretrained model
model_path = 'C:/Users/green/Desktop/my code AI/RNN11model.pth'
input_size = 13  # Number of MFCC features
hidden_size = 64
num_layers = 2
num_classes = len([ 'sad', 'notsad'])  # Define your emotion labels

emotion_model = EmotionRNN(input_size, hidden_size, num_layers, num_classes)
emotion_model.load_state_dict(torch.load(model_path))
emotion_model.eval()



# Initialize the label encoder
emotions = [ 'sad', 'notsad']
label_encoder = LabelEncoder()
label_encoder.fit(emotions)

# Load a random .wav audio file and extract features
audio_file_path = 'sad11.wav'  # Replace with the path to your audio file
features = extract_features(audio_file_path, max_length=100)

if features is not None:
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        logits = emotion_model(features)
        predicted_label_idx = torch.argmax(logits, dim=1).cpu().numpy()[0]
        predicted_emotion = label_encoder.inverse_transform([predicted_label_idx])[0]

    print(f'Predicted Emotion: {predicted_emotion}')
else:
    print('Failed to extract features from the audio file.')


