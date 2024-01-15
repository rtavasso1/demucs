import os
import sys
import torch

from demucs.htdemucs import HTDemucs
import torchaudio

model = HTDemucs(sources=['Drums'], audio_channels=1)
model = model.to('cuda')

data, sr = torchaudio.load('test.mp3') # 20 seconds of data
data = data[None, ...].to('cuda')
# repeat to make it longer
data = data.repeat(1, 1, 1) # batch of 64 of 20 seconds
data = data.mean(1, keepdim=True) # mono

out = model(data)
print(out.shape)
loss = out.mean()
loss.backward()

# Show VRAM usage
print(torch.cuda.memory_summary())