"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from data import AudioMIDIDataset
from torch.utils.data import DataLoader
from demucs.htdemucs import HTDemucs
from metrics import f1_score
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume''
eval_only = False # if True, only run eval loop
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'Automatic Drum Transcription'
wandb_run_name = 'HTDemucs' + str(time.time()) # 'run' + str(time.time())
# data
dataset = 'e-gmd-v1.0.0'
gradient_accumulation_steps = 8 # used to simulate larger batch sizes (HTDemucs uses batch size of 32)
batch_size = 2 # if gradient_accumulation_steps > 1, this is the micro-batch size
sample_rate = 44100 # Hz
sample_length = 20 # s
block_size = sample_rate * sample_length # number of samples in a block
# model
sources=[str(i) for i in range(88)]
audio_channels=1
depth=4
t_layers=5
t_heads=8
t_dropout = 0.0
t_weight_decay=0.0
samplerate=sample_rate
segment=sample_length, 
velocity_branch=True
# adamw optimizer
learning_rate = 3e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 0.05
beta1 = 0.9
beta2 = 0.999
grad_clip = 5.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 3e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# loss function
bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
regression_loss = torch.nn.L2Loss()
velocity_loss_coeff = 0.5 # coefficient for velocity loss
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16' # 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
print('gpu available: ', torch.cuda.is_available(), 'dtype: ', dtype)
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
# tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
# print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
train_data = AudioMIDIDataset('../data/e-gmd-v1.0.0', sample_length=sample_length, eval=False)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=1)
val_data = AudioMIDIDataset('../data/e-gmd-v1.0.0', sample_length=sample_length, eval=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=1)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(sources=sources, audio_channels=audio_channels, depth=depth, t_layers=t_layers, t_heads=t_heads,
                   t_dropout=t_dropout, t_weight_decay=t_weight_decay, samplerate=sample_rate, segment=sample_length, 
                   velocity_branch=velocity_branch) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    model = HTDemucs(**model_args)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    model = HTDemucs(**model_args)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('HTDemucs'):
    print(f"Initializing from pretrained weights: {init_from}")
    model = HTDemucs(**model_args)
    # TODO: load the pretrained weights

    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['sources', 'audio_channels', 'depth', 't_layers', 't_heads', 't_dropout', 't_weight_decay', 'samplerate', 'segment', 'velocity_branch']:
        model_args[k] = getattr(model, k)

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for loader, split in zip([train_dataloader, val_dataloader],[ 'train', 'val']):
        num_iters = min(eval_iters, len(loader))
        losses = torch.zeros(num_iters)
        f1_scores = torch.zeros(num_iters)
        f1_w_vel_scores = torch.zeros(num_iters)
        iters = 0
        for batch in loader:
            if iters >= eval_iters:
                break
            waveform, onset, velocity = batch
            
            waveform = waveform.to(device, dtype=ptdtype)
            onset = onset.to(device, dtype=ptdtype)
            velocity = velocity.to(device, dtype=ptdtype)

            with ctx:
                onset_pred, velocity_pred = model(waveform)
                print(onset_pred.device,onset.device)
                loss = bce_loss(onset_pred, onset) + velocity_loss_coeff * regression_loss(velocity_pred, velocity)
            f1, f1_w_vel = f1_score(onset, onset_pred, velocity, velocity_pred)
            losses[iters] = loss.item()
            f1_scores[iters] = f1
            f1_w_vel_scores[iters] = f1_w_vel
            iters += 1
        
        out[split] = {'Loss': losses.mean(), 'F1': f1_scores.mean(), 'F1 w/ Vel': f1_w_vel_scores.mean()}
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:
    for batch in train_dataloader:
        audio, onsets, velocities = batch
        #print('audio shape: ', audio.device, audio.dtype)
        audio = audio.to(device, dtype=ptdtype)
        #print('audio shape: ', audio.device, audio.dtype)
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']['Loss']:.4f}, val loss {losses['val']['Loss']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train']['Loss'],
                    "val/loss": losses['val']['Loss'],
                    "train/f1": losses['train']['F1'],
                    "val/f1": losses['val']['F1'],
                    "train/f1_w_vel": losses['train']['F1 w/ Vel'],
                    "val/f1_w_vel": losses['val']['F1 w/ Vel'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
            if losses['val']['Loss'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']['Loss']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                pred_onsets, pred_vel = model(audio)
                loss = bce_loss(pred_onsets, onsets) + velocity_loss_coeff * regression_loss(pred_vel, velocities)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()