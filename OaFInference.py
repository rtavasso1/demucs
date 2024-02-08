import torch
from model import OnsetsAndFrames
from torchaudio.transforms import MelSpectrogram
import torchaudio
import pretty_midi
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data import AudioMIDIDataset
from metrics import f1_score

SAMPLE_RATE = 44100
HOP_LENGTH = 441
N_MELS = 250
WINDOW_LENGTH = 2048
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2

def load_model(file, LSTM_size, device='cuda'):
    ckpt_path = os.path.join('out', file)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model']
    unwanted_prefix = 'module.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model = OnsetsAndFrames(LSTM_size=LSTM_size)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def transcribe(files, threshold=0.5):
    for file in files:
        # Initialize PrettyMIDI object
        midi_file = pretty_midi.PrettyMIDI()

        # Initialize an Instrument instance for a drum kit (channel 9 is for drums)
        instrument = pretty_midi.Instrument(program=9, is_drum=True)

        x, sr = torchaudio.load(file)
        x = x.sum(0, keepdim=True).unsqueeze(0) # (B, C, T)
        x = torch.clamp(x, -1, 1)
        mel = melspectrogram(x)
        mel = mel.squeeze(1).transpose(1, 2) # (B, T, F)
        onset, vel = model(mel)
        
        # Convert to MIDI file and save
        onset = torch.sigmoid(onset).squeeze().detach().cpu().numpy()
        vel = vel.squeeze().detach().cpu().numpy()
        onset = onset > threshold
        onset = onset.astype(int)
        vel = vel/np.max(vel) * 80 + 10
        vel = np.clip(vel, 0, 127)
        midi_path = file.replace('.mp3', '.mid')

        for i, (o_i, v_i) in enumerate(zip(onset, vel)):
            for j, (o_ij, v_ij) in enumerate(zip(o_i, v_i)):
                if o_ij:
                    instrument.notes.append(pretty_midi.Note(
                        velocity=int(v_ij),
                        pitch=j,  # Example: Closed Hi-Hat in General MIDI
                        start=i * HOP_LENGTH / SAMPLE_RATE,
                        end=(i + 1) * HOP_LENGTH / SAMPLE_RATE
                    ))

        # Add the instrument to the PrettyMIDI object
        midi_file.instruments.append(instrument)

        # Write out the MIDI data to a new file
        midi_file.write(midi_path)

@torch.no_grad()
def get_metrics(num_batches=20, threshold=0.5, step_size=0.05):
    dataset = AudioMIDIDataset('K:\\dataset\\e-gmd-v1.0.0\\e-gmd-v1.0.0', sample_length=12, OaF_midi=True, eval=False) # Adjust num_samples as needed
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    if isinstance(threshold, tuple):
        threshold = np.linspace(*threshold, int((threshold[1]-threshold[0])/step_size)+1)
    else:
        threshold = [threshold]

    f1_scores = torch.zeros(num_batches, len(threshold))
    f1_w_vel_scores = torch.zeros(num_batches, len(threshold))

    for i, batch in enumerate(dataloader):
        if i == num_batches:
            break
        audio, onset, velocity = batch
        audio = torch.clamp(audio, -1, 1)
        mel = melspectrogram(audio)
        mel = mel.squeeze(1).transpose(1, 2)
        onset_pred, velocity_pred = model(mel)

        for j, t in enumerate(threshold):
            f1, f1_w_vel = f1_score(onset, onset_pred, velocity, velocity_pred, OaF_midi=True, threshold=t)
            f1_scores[i,j] = f1
            f1_w_vel_scores[i,j] = f1_w_vel

    for j,t in enumerate(threshold):
        print(f"Threshold: {t} F1: {f1_scores[:,j].mean():.3f}, F1 w/ Vel: {f1_w_vel_scores[:,j].mean():.3f}")

model_paths = ['ckpt_OaF_32k.pt', 'ckpt_OaF_128LSTM_Shuffle_step6.pt', 
               'ckpt_OaF_128LSTM_Shuffle_Unweighted_20s_step16000.pt', 'ckpt_OaF_Transformer_Shuffle_Unweighted_20s_step10000.pt']
model_path = model_paths[3]
try:
    model = load_model(model_path, LSTM_size=64, device='cpu')
except:
    model = load_model(model_path, LSTM_size=32, device='cpu')
melspectrogram = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=WINDOW_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS, center=False, f_min=MEL_FMIN, f_max=MEL_FMAX, pad=(WINDOW_LENGTH-HOP_LENGTH)//2+1)

files = ['.\\mp3s\\Coast.mp3drums.mp3']
model.train()
#transcribe(files, threshold=0.15)
get_metrics(num_batches=20, threshold=(0.1, 0.2), step_size=0.02)