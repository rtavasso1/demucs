import torch
import mir_eval
import numpy as np
import warnings

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

def extract_notes(onsets, velocities, onset_threshold=0.5, predictions=False):
    """
    Finds the onset timings and corresponding velocity values based on the onsets information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float

    Returns
    -------
    sorted_onset_indices: np.ndarray of onset frame indices sorted chronologically
    sorted_velocities: np.ndarray of velocity values for each onset sorted chronologically
    """
    if predictions:
        onsets = torch.sigmoid(onsets)
        onsets = onsets > onset_threshold
        velocities = torch.clamp(velocities, 0, 127)

    onset_vel_pairs = []

    # Iterate over each frame and bin
    for frame in range(onsets.shape[0]):
        for pitch in range(onsets.shape[1]):
            if onsets[frame, pitch].item() == 1:
                onset_vel_pairs.append((frame, velocities[frame, pitch].item()))

    # Sort by frame index (chronological order)
    onset_vel_pairs.sort(key=lambda x: x[0])

    # Separate the sorted pairs into two lists
    sorted_onset_indices, sorted_velocities = zip(*onset_vel_pairs)

    return np.array(sorted_onset_indices), np.array(sorted_velocities)

def sparse_to_dense(onsets, velocities, sample_rate=44100, predictions=False):
    """
    Convert sparse MIDI data to dense MIDI data.

    Args:
        onsets: (num_samples,88) bool tensor of onsets
        velocities: (num_samples,) tensor of velocities
        sample_rate: sample rate of the audio file

    Returns:
        dense_onsets: (num_onsets,) tensor of onset times in seconds
        dense_velocities: (num_onsets,) tensor of velocities
    """
    if predictions:
        onsets = torch.sigmoid(onsets)
        onsets = onsets > 0.5
        velocities = torch.clamp(velocities, 0, 127)

    onsets_mask = onsets.any(dim=-1)
    dense_onsets = torch.nonzero(onsets_mask, as_tuple=False).squeeze()[:,1] / sample_rate
    dense_velocities = velocities[onsets_mask]

    return dense_onsets, dense_velocities

def f1_score(sparse_reference_onsets, sparse_estimated_onsets, sparse_reference_velocities, sparse_estimated_velocities, OaF_midi=False):
    """
    Calculate F1 scores w/ and w/o velocities

    Args:
        reference_onsets: (num_samples, 88) tensor of onsets
        estimated_onsets: (num_samples, 88) tensor of onsets
        reference_velocities: (num_samples,) tensor of velocities
        estimated_velocities: (num_samples,) tensor of velocities

    Returns:
        f1_score: F1 score
        f1_score_no_vel: F1 score without considering velocities
    """
    if not OaF_midi:
        estimated_onsets, estimated_velocities = sparse_to_dense(estimated_onsets, estimated_velocities, predictions=True) # (num_onsets,), (num_onsets,)
        reference_onsets, reference_velocities = sparse_to_dense(reference_onsets, reference_velocities) # (num_onsets,), (num_onsets,)

        reference_onsets = reference_onsets.cpu().numpy().astype(np.float32)
        estimated_onsets = estimated_onsets.detach().cpu().numpy().astype(np.float32)
        reference_velocities = reference_velocities.cpu().numpy().astype(np.float32)
        estimated_velocities = estimated_velocities.detach().cpu().numpy().astype(np.float32)

        # sort onsets in ascending order and sort velocities accordingly
        sort_indices = reference_onsets.argsort()
        reference_onsets = reference_onsets[sort_indices]
        reference_velocities = reference_velocities[sort_indices]

        sort_indices = estimated_onsets.argsort()
        estimated_onsets = estimated_onsets[sort_indices]
        estimated_velocities = estimated_velocities[sort_indices]
    else:
        batch_f1_scores = []
        batch_f1_scores_w_vel = []
        for i in range(sparse_reference_onsets.shape[0]):
            estimated_onsets, estimated_velocities = extract_notes(sparse_estimated_onsets[i], sparse_estimated_velocities[i], predictions=True) # (num_onsets,), (num_onsets,)
            reference_onsets, reference_velocities = extract_notes(sparse_reference_onsets[i], sparse_reference_velocities[i]) # (num_onsets,), (num_onsets,)

            f1_score = mir_eval.onset.f_measure(reference_onsets, estimated_onsets)

            # Create reference intervals by stacking onsets into 2 columns, reference pitches is an array of zeros
            reference_intervals = np.stack((reference_onsets, reference_onsets+0.001)).T
            reference_pitches = np.ones_like(reference_velocities)

            # Create estimated intervals by stacking onsets into 2 columns, estimated pitches is an array of zeros
            estimated_intervals = np.stack((estimated_onsets, estimated_onsets+0.001)).T
            estimated_pitches = np.ones_like(estimated_velocities)

            f1_score_w_vel = mir_eval.transcription_velocity.precision_recall_f1_overlap(reference_intervals, reference_pitches, reference_velocities, estimated_intervals, estimated_pitches, estimated_velocities)
            batch_f1_scores.append(f1_score[0])
            batch_f1_scores_w_vel.append(f1_score_w_vel[2])

    return np.mean(batch_f1_scores), np.mean(batch_f1_scores_w_vel)