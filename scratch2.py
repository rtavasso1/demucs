#
import torch
from model import OnsetsAndFrames

SAMPLE_RATE = 44100
HOP_LENGTH = 441
N_MELS = 250
WINDOW_LENGTH = 2048
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2

# 

model = OnsetsAndFrames()

x = torch.randn(8, 1, 44100*12)
x = torch.clamp(x, -1, 1)

if True: # pt
    from torchaudio.transforms import MelSpectrogram
    melspectrogram = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=WINDOW_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS, center=False, f_min=MEL_FMIN, f_max=MEL_FMAX, pad=(WINDOW_LENGTH-HOP_LENGTH)//2+1)
    mel = melspectrogram(x)
    mel = mel.squeeze(1).transpose(1, 2)
    onset, vel = model(mel)
elif False: # oaf
    from model import MelSpectrogram
    melspectrogram = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX)
    mel = melspectrogram(x.squeeze())
    onset, vel = model(mel)
else:
    from librosa.feature import melspectrogram
    mel = melspectrogram(y=x.squeeze().numpy(), sr=SAMPLE_RATE, n_fft=WINDOW_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX, center=False)
    mel = torch.from_numpy(mel)
    mel = mel.transpose(-1,-2)
    print(mel.shape)
    onset, vel = model(mel)

print(mel.shape, onset.shape, vel.shape)
