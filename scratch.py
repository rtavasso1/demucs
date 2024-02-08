import torch
from torch.utils.data import Dataset, DataLoader

from data import AudioMIDIDataset, shuffled_mixup
from model import OnsetsAndFrames as OaF, MelSpectrogram
from metrics import sparse_to_dense, f1_score

import pretty_midi
import matplotlib.pyplot as plt
import numpy as np

def plot_midi_and_onsets(midi_path, onsets, sample_rate=44100):
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    # for note in midi_data.instruments[0].notes:
    #     print(f"Note Pitch: {note.pitch}, Start: {note.start}, End: {note.end}")

    # Plot the original MIDI notes
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    for note in midi_data.instruments[0].notes:
        ax[0].barh(note.pitch, note.end - note.start, left=note.start, height=1, align='center')
    ax[0].set_title('Original MIDI Notes')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Pitch')

    # Convert onsets to a plot-friendly format
    onset_times, pitches = onsets.nonzero(as_tuple=True)
    onset_times = onset_times / 100  # Convert back to seconds

    # for time, pitch in zip(onset_times, pitches):
    #     print(f"Processed Note Pitch: {pitch}, Time: {time}")

    # Plot the processed onsets
    ax[1].barh(pitches, 0.05, left=onset_times, height=1, align='center')
    ax[1].set_title('Processed Onsets')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Pitch')

    plt.tight_layout()
    # plt.savefig('midi_and_onsets.png')
    plt.show()


# Usage
dataset = AudioMIDIDataset('K:\\dataset\\e-gmd-v1.0.0\\e-gmd-v1.0.0', sample_length=12, OaF_midi=True, eval=False) # Adjust num_samples as needed
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = OaF()

# Processor
SAMPLE_RATE = 44100
HOP_LENGTH = 441 # SAMPLE_RATE * 32 // 1000
N_MELS = 250
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 2048
melspectrogram = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX)

#audio, onsets, velocities, midi_path = next(iter(dataloader))

# Plot the MIDI and onsets
#plot_midi_and_onsets(midi_path[0], onsets[0])


# Initialize accumulators for positive and negative counts for each note
# Assuming the shape of onsets in each batch is [B, frames, 88]
accumulated_positives = torch.zeros(88)
accumulated_negatives = torch.zeros(88)

# Assuming 'dataloader' is your DataLoader instance
iteration_count = 0
for batch in dataloader:
    iteration_count += 1
    audio, onsets, _ = batch  # Assuming 'onsets' has the shape [B, frames, 88]

    # Update positive and negative counts
    positive_counts = torch.sum(onsets, dim=[0, 1])  # Sum over batch and frames, resulting in shape [88]
    total_counts = onsets.shape[1] * onsets.shape[0]  # Total count per note
    negative_counts = total_counts - positive_counts

    accumulated_positives += positive_counts
    accumulated_negatives += negative_counts

    # Every 500 iterations, compute and print the mean class imbalance for each note
    if iteration_count % 500 == 0:
        # Avoid division by zero
        safe_positive_counts = torch.where(accumulated_positives == 0, torch.ones_like(accumulated_positives), accumulated_positives)
        mean_class_imbalance = accumulated_negatives / safe_positive_counts
        print(mean_class_imbalance)


# Note: Ensure 'shuffled_mixup' preserves the shape and structure of 'onsets'




    # mel = melspectrogram(audio.squeeze())
    # print('inputs shape: ', audio.shape, onsets.shape, velocities.shape, mel.shape)
    
    # pred_onsets, pred_vel = model(mel)
    # print('pred shape: ', pred_vel.shape, pred_onsets.shape)
    # break
    # #dense_onsets, dense_velocities = sparse_to_dense(pred_onsets, pred_vel, predictions=True)
    # #dense_onsets, dense_velocities = extract_notes(pred_onsets, pred_vel)
    # f1, f1_w_vel = f1_score(onsets, pred_onsets, velocities, pred_vel, OaF_midi=True)
    # print(f1, f1_w_vel)
    # break