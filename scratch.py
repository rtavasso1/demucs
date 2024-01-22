import torch
from torch.utils.data import Dataset, DataLoader

from data import AudioMIDIDataset, shuffled_mixup
from model import OnsetsAndFrames as OaF, MelSpectrogram
from metrics import sparse_to_dense, f1_score

# Usage
dataset = AudioMIDIDataset('../data/e-gmd-v1.0.0', sample_length=5, OaF_midi=True) # Adjust num_samples as needed
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = OaF()

# Processor
SAMPLE_RATE = 44100
HOP_LENGTH = 441 # SAMPLE_RATE * 32 // 1000
N_MELS = 250
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 2048
melspectrogram = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX)

# Training loop
for batch in dataloader:
    audio, onsets, velocities = batch
    audio, onsets, velocities = shuffled_mixup(batch, 5*44100, 5)
    mel = melspectrogram(audio.squeeze())
    print('inputs shape: ', audio.shape, onsets.shape, velocities.shape, mel.shape)
    
    pred_onsets, pred_vel = model(mel)
    print('pred shape: ', pred_vel.shape, pred_onsets.shape)
    #dense_onsets, dense_velocities = sparse_to_dense(pred_onsets, pred_vel, predictions=True)
    #dense_onsets, dense_velocities = extract_notes(pred_onsets, pred_vel)
    f1, f1_w_vel = f1_score(onsets, pred_onsets, velocities, pred_vel, OaF_midi=True)
    print(f1, f1_w_vel)
    break