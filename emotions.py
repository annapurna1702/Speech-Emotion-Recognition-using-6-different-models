import torch # type: ignore
import librosa # type: ignore
from nemo.collections.asr.models import EncDecClassificationModel # type: ignore

# Load your .pt model file using PyTorch
pt_model_path = "C:/Users/green/Desktop/my code AI/models/audio/audio_model.pth"
emotion_labels = ["happy","sad", "angry"]
loaded_model = torch.load(pt_model_path, map_location='cpu')

# Define a custom model wrapper class
class CustomModelWrapper:
    def __init__(self, model):
        self.model = model
    
    def transcribe(self, audio_data_tensor):
        # Perform inference using the loaded model
        # Ensure that the model's output matches the expected format
        with torch.no_grad():
            output = self.model(audio_data_tensor)
        return output  # Return results in the expected format

# Instantiate the custom model wrapper
model = CustomModelWrapper(loaded_model)

audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
audio_data_list = []
sample_rate = None  # Initialize sample rate

for audio_file in audio_files:
    # Load audio data from file
    audio_data, sr = librosa.load(audio_file, sr=None)

    # If this is the first file, set the sample rate
    if sample_rate is None:
        sample_rate = sr

    # Convert audio data to a PyTorch tensor and ensure mono (single channel)
    audio_data_tensor = torch.tensor(audio_data).float()
    if len(audio_data_tensor.shape) > 1:
        audio_data_tensor = audio_data_tensor.mean(dim=0)  # Convert to mono if stereo

    # Append the audio data tensor to the list
    audio_data_list.append(audio_data_tensor)

# Ensure all audio data tensors are the same length
min_length = min(len(tensor) for tensor in audio_data_list)
audio_data_list = [tensor[:min_length] for tensor in audio_data_list]

# Stack the three audio data tensors along the channel dimension
audio_data_tensor = torch.stack(audio_data_list, dim=0)

# Add a batch dimension
audio_data_tensor = audio_data_tensor.unsqueeze(0)

# Reshape the audio data tensor to match the expected input shape of the model
audio_data_tensor = audio_data_tensor.view(1, 3, 1, min_length)

# Perform inference with the model
results = model.transcribe(audio_data_tensor)
for prediction in results:
    # Use torch.argmax to find the index of the highest predicted value
    predicted_index = torch.argmax(prediction).item()

    # Map the predicted index to the corresponding emotion label
    predicted_emotion = emotion_labels[predicted_index]

    # Print the predicted emotion
print("Predicted emotion:", predicted_emotion)

# Print the results
#print("Predicted results:", results)
# Load the audio file using librosa












'''
audio_file = "audio.wav"
audio_data, sample_rate = librosa.load(audio_file, sr=None)

# Convert audio data to a tensor
audio_data_tensor = torch.tensor(audio_data).unsqueeze(0).float()

# Assume model expects input in the format [batch_size, channels, height, width]
# Reshape the tensor from [1, 112320] to [1, 1, height, width] based on your model's expected input
# This reshaping assumes mono audio with a height of 1 and width equal to the length of the audio_data
height = 1
width = audio_data_tensor.shape[1] // height
audio_data_tensor = audio_data_tensor.view(1, height, width)

# Perform inference
results = model.transcribe(audio_data_tensor)

# The results contain the predicted classifications
print("Predicted results:", results)


# Load the audio file using librosa'''
'''
audio_file = "audio.wav"
# Load the audio file using librosa
audio_file = "audio.wav"
audio_data, sample_rate = librosa.load(audio_file, sr=None)

# Convert audio data to a tensor
audio_data_tensor = torch.tensor(audio_data).unsqueeze(0).float()

# Assume model expects input in the format [batch_size, channels, height, width]
# Reshape the tensor from [1, 112320] to [1, 1, height, width] based on your model's expected input
# This reshaping assumes mono audio with a height of 1 and width equal to the length of the audio_data
height = 1
width = audio_data_tensor.shape[1] // height
audio_data_tensor = audio_data_tensor.view(1, height, width)

# Perform inference
results = model.transcribe(audio_data_tensor)

# The results contain the predicted classifications
print("Predicted results:", results)'''
'''
audio_data, sample_rate = librosa.load(audio_file, sr=None)

# Ensure the audio data is mono and convert to tensor
audio_data = librosa.to_mono(audio_data)
audio_data_tensor = torch.tensor(audio_data).unsqueeze(0).float()

# Perform inference using the custom model
results = model.transcribe(audio_data_tensor)

# Print the predicted classifications
print("Predicted results:", results)
'''