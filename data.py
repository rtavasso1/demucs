import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
import torchaudio
import pathlib
import pretty_midi
import numpy as np

class AudioMIDIDataset(Dataset):
    def __init__(self, directory, sample_length, sample_rate=44100, eval=False, dense_representation=False, OaF_midi=False):
        self.directory = pathlib.Path(directory)
        self.num_samples = sample_length * sample_rate
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        self.dense_representation = dense_representation
        self.data = []
        self.OaF_midi = OaF_midi
        for path in self.directory.rglob('*.wav'):
            if eval:
                if 'eval' not in str(path):
                    continue
            else:
                if 'eval' in str(path):
                    continue
            midi_path = path.with_suffix('.midi')
            if midi_path.exists():  # Check if MIDI file exists
                self.data.append((str(path), str(midi_path)))

        if len(self.data) == 0:
            print("No data found. Check the dataset directory path and contents.")
            print(self.directory)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, midi_path = self.data[idx]

        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform_length = waveform.size(1)

        # Initialize start and end for slicing the waveform
        start, end = 0, waveform_length

        # Trim or pad the waveform to a fixed size
        if waveform_length >= self.num_samples:
            # Randomly select a start point for trimming
            start = random.randint(0, waveform_length - self.num_samples)
            end = start + self.num_samples
            waveform = waveform[:, start:end]
        else:
            # Pad with zeros if the file is shorter than num_samples
            padding = self.num_samples - waveform_length
            waveform = torch.nn.functional.pad(waveform, (0, padding), "constant", 0)
        #waveform = waveform.to(torch.bfloat16)
        # Load MIDI data
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            print(f"Error loading MIDI file {midi_path}: {e}")
            return None  # Or handle the error as you see fit

        if False: # used for generating dense representations, but instead I just process the sparse -> dense for mir_eval
            onsets, velocities = [], []
            for note in midi_data.instruments[0].notes:
                start_sample = int(note.start * sample_rate) - start
                if 0 <= start_sample < self.num_samples:
                    onsets.append(start_sample)
                    velocities.append(start_sample)

        if self.OaF_midi:
            onsets_sparse = np.zeros((self.sample_length * 100, 88))
            velocities_sparse = np.zeros((self.sample_length * 100, 88))

            for note in midi_data.instruments[0].notes:
                start_sample = (int(note.start * sample_rate) - start) * 100 // sample_rate # 100 is the 'sample rate' of the mel spectrogram
                if 0 <= start_sample < (self.num_samples * 100 // sample_rate):
                    onsets_sparse[start_sample, note.pitch] = 1
                    velocities_sparse[start_sample, note.pitch] = note.velocity
            # Convert arrays to tensors
            onsets_sparse = torch.tensor(onsets_sparse, dtype=torch.float32) # (num_samples,88)
            velocities_sparse = torch.tensor(velocities_sparse, dtype=torch.float32) # (num_samples,88)
        else:      
            # Initialize arrays for onsets and velocities
            onsets_sparse = np.zeros((self.num_samples,88))
            velocities_sparse = np.zeros((self.num_samples,))

            # Fill in the onsets and velocities arrays
            for note in midi_data.instruments[0].notes:
                start_sample = int(note.start * sample_rate) - start
                if 0 <= start_sample < self.num_samples:
                    onsets_sparse[start_sample, note.pitch] = 1
                    velocities_sparse[start_sample] = note.velocity
            # Convert arrays to tensors
            onsets_sparse = torch.tensor(onsets_sparse, dtype=torch.float32) # (num_samples,88)
            velocities_sparse = torch.tensor(velocities_sparse, dtype=torch.float32) # (num_samples,88)

        return waveform, onsets_sparse, velocities_sparse

def deranged_shuffle(tensor):
    cloned_tensor = tensor.clone()
    # Perform a Fisher-Yates shuffle with derangement
    for i in range(tensor.size(0) - 1, 0, -1):
        # Generate a random index that is not equal to i
        j = torch.randint(0, i, (1,))
        while j == i:
            j = torch.randint(0, i, (1,))
        
        # Swap elements
        tensor[i], tensor[j] = cloned_tensor[j], cloned_tensor[i]
    return tensor

def mixup(waveform, onsets, velocities):
        # Randomly create pairwise relationships between samples within a batch and mix (combine/add) them
        batch_size = waveform.size(0)
        indices = torch.arange(batch_size)
        indices = deranged_shuffle(indices)

        shuffled_waveform = waveform[indices]
        shuffled_onsets = onsets[indices]
        shuffled_velocities = velocities[indices]

        combined_waveform = waveform + shuffled_waveform
        combined_onsets = onsets + shuffled_onsets
        combined_velocities = velocities + shuffled_velocities

        return combined_waveform, combined_onsets, combined_velocities

def shuffle(waveform, onsets, velocities, num_samples, sample_length, sr_ratio=441):
    # sr_ratio is the ratio of the sample rate of 'waveform' to that of 'onsets' and 'velocities'
    batch_size = waveform.size(0)

    # Chunk the waveform
    chunked_waveform = waveform.view(sample_length * batch_size, 1, num_samples // sample_length)

    # Adjust the chunking for onsets and velocities based on the sample rate ratio
    onsets_sample_length = int(num_samples // sr_ratio)
    velocities_sample_length = int(num_samples // sr_ratio)

    chunked_onsets = onsets.view(sample_length * batch_size, onsets_sample_length // sample_length, 88)
    chunked_velocities = velocities.view(sample_length * batch_size, velocities_sample_length // sample_length, 88)

    # Shuffle
    indices = torch.randperm(batch_size * sample_length)
    shuffled_waveform = chunked_waveform[indices]
    shuffled_onsets = chunked_onsets[indices]
    shuffled_velocities = chunked_velocities[indices]

    # Reassemble
    combined_waveform = shuffled_waveform.view(batch_size, 1, num_samples)
    combined_onsets = shuffled_onsets.view(batch_size, onsets_sample_length, 88)
    combined_velocities = shuffled_velocities.view(batch_size, velocities_sample_length, 88)

    return combined_waveform, combined_onsets, combined_velocities


def shuffled_mixup(batch, num_samples, sample_length):
    # Separate the batch into individual components
    waveforms, onsets, velocities = batch

    # Apply mixup and shuffle
    waveforms, onsets, velocities = mixup(waveforms, onsets, velocities)
    waveforms, onsets, velocities = shuffle(waveforms, onsets, velocities, num_samples, sample_length)

    return waveforms, onsets, velocities

# # Usage
# dataset = AudioMIDIDataset('../data/e-gmd-v1.0.0', sample_length=10) # Adjust sample_length as needed
# dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# # Training loop
# from metrics import sparse_to_dense, f1_score
# from demucs.htdemucs import HTDemucs
# # model = HTDemucs(sources=[str(i) for i in range(88)], audio_channels=1).to('cuda:0')
# imbalances = []
# for batch in dataloader:
#     if batch is not None:
#         audio, onsets, velocities = batch
#         print(audio.shape, onsets.shape, velocities.shape)
#         num_positive = torch.sum(onsets) # (batch_size,)
#         num_negative = torch.sum(1 - onsets) # (batch_size,)
#         class_imbalance = num_negative / num_positive # (batch_size,)
#         print(class_imbalance)
#         imbalances.append(class_imbalance)
#         if len(imbalances) > 100:
#             break
#         #break
# print(np.mean(imbalances))
# #         #shuffled_audio, shuffled_onsets, shuffled_velocities = shuffled_mixup(batch)
# #         #print(shuffled_audio.shape, shuffled_onsets.shape, shuffled_velocities.shape)

#         audio = audio.to('cuda:0')
#         print(audio.dtype)
#         pred_onsets, pred_vel = model(audio)

#         f1, f1_w_vel = f1_score(onsets, pred_onsets, velocities, pred_vel)
#         print(f1, f1_w_vel)

#         break
#         # Your training code here
#     else:
#         print("Encountered a problem with a data item.")
#         continue
