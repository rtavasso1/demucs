import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from librosa.filters import mel
from librosa.util import pad_center
from scipy.signal import get_window
from torch.autograd import Variable
import inspect
from rotary_embedding_torch import RotaryEmbedding

SAMPLE_RATE = 44100
HOP_LENGTH = 441 # SAMPLE_RATE * 32 // 1000
N_MELS = 250
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 2048

class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length, hop_length, win_length=None, window='hann'):
        super(STFT, self).__init__()
        if win_length is None:
            win_length = filter_length

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())

    def forward(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase


class MelSpectrogram(torch.nn.Module):
    def __init__(self, n_mels, sample_rate, filter_length, hop_length,
                 win_length=None, mel_fmin=0.0, mel_fmax=None):
        super(MelSpectrogram, self).__init__()
        self.stft = STFT(filter_length, hop_length, win_length)

        mel_basis = mel(sr=sample_rate, n_fft=filter_length, n_mels=n_mels, fmin=mel_fmin, fmax=mel_fmax, htk=True)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def forward(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, T, n_mels)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)
        #print('mel forward called')
        magnitudes, phases = self.stft(y)
        #print('stft done')
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = torch.log(torch.clamp(mel_output, min=1e-5))
        mel_output = mel_output.transpose(1, 2)[:,:-1,:]
        return mel_output

class BiLSTM(nn.Module):
    inference_chunk_length = 512

    def __init__(self, input_features, recurrent_features):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=True)

    def forward(self, x):
        if self.training:
            print(x.shape)
            return self.rnn(x)[0]
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

            # forward direction
            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.rnn.bidirectional:
                h.zero_()
                c.zero_()

                for start in reversed(slices):
                    end = start + self.inference_chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

            return output
class CustomAttention(nn.Module):
    def __init__(self, d_model, n_heads, rotary_emb):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(dim=d_model // n_heads // 2)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Linear layers for projecting the input to Q, K, V spaces
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, self.n_heads, 3 * self.d_head).permute(2, 0, 3, 1)
        q, k, v = qkv.chunk(3, dim=-2)

        # Apply rotary embeddings to queries and keys
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # Compute attention (scaled dot-product attention)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v).transpose(1, 2).reshape(batch_size, seq_len, -1)

        return attn_output

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(dim=d_model // nhead)
        self.self_attn = CustomAttention(d_model, nhead, self.rotary_emb)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(self.norm1(src))
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2)
        return src

class MyTransformer(nn.Module):
    def __init__(self, d_model, n_head, n_layers, dropout):
        super().__init__()
        self.encoder = nn.ModuleList([CustomTransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, dim_feedforward=d_model*4) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x
class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            #nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Linear(1984, output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        #print('mel shape: ', mel.shape)
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        #print('x shape: ', x.shape)
        x = self.cnn(x)
        #print('cnn shape: ', x.shape)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features=250, output_features=88, model_size=256, LSTM_size=64):
        super().__init__()

        # sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size)
        sequence_model = lambda a, b: MyTransformer(d_model=model_size, n_head=4, n_layers=4, dropout=0.1)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, LSTM_size),
            nn.Dropout(0.5),
            # nn.Linear(LSTM_size * 2, output_features)
            nn.Linear(model_size, output_features)
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        velocity_pred = self.velocity_stack(mel)

        return onset_pred, velocity_pred
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        use_fused = False
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer