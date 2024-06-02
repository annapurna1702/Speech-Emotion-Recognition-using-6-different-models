import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp

from nemo.collections.asr.models import EncDecClassificationModel

import numpy as np
from nemo.utils import audio

# Load the pre-trained model (you can specify the path or use the default model)
model = EncDecClassificationModel.from_pretrained("path_to_pretrained_model_or_default")



# Load the audio file
audio_file = "audio.wav"
audio_data = audio.read(audio_file)
# Perform inference
results = model.transcribe([audio_data])

# The results contain the predicted emotions
print("Predicted emotions:", results)
