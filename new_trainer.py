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
def main():
    import os
    import time
    import math
    import pickle
    from contextlib import nullcontext

    import numpy as np
    import torch
    from accelerate import Accelerator
    from torch.distributed import init_process_group, destroy_process_group

    from data import AudioMIDIDataset, shuffled_mixup
    from torch.utils.data import DataLoader
    from demucs.htdemucs import HTDemucs
    from model import OnsetsAndFrames # , MelSpectrogram
    from torchaudio.transforms import MelSpectrogram
    from metrics import f1_score
    # -----------------------------------------------------------------------------
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    out_dir = 'out'
    eval_interval = 2000
    log_interval = 10
    eval_iters = 20
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    init_from = 'scratch' # 'scratch' or 'resume''
    eval_only = False # if True, only run eval loop
    model_name = 'OaF' # 'OaF' or 'HTDemucs'
    # wandb logging
    wandb_log = True # disabled by default
    wandb_project = 'Automatic Drum Transcription'
    wandb_run_name = model_name + str(time.time()) # 'run' + str(time.time())
    # data
    dataset = 'e-gmd-v1.0.0'
    gradient_accumulation_steps = 1 # used to simulate larger batch sizes (HTDemucs uses batch size of 32)
    batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
    sample_rate = 44100 # Hz
    sample_length = 20 # s
    num_samples = sample_rate * sample_length # number of samples in a block
    # HTDemucs model
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
    learning_rate = 1e-4 # max learning rate
    max_iters = 600000 # total number of training iterations
    weight_decay = 0.00 # 0.05
    beta1 = 0.9
    beta2 = 0.999
    grad_clip = 3.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
    min_lr = 3e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # loss function
    velocity_loss_coeff = 0.5 # coefficient for velocity loss
    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'float32' # 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    print('gpu available: ', torch.cuda.is_available(), 'dtype: ', dtype)
    compile = True # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------


    accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=gradient_accumulation_steps)
    master_process = accelerator.is_main_process

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    # torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    # poor man's data loader
    train_data = AudioMIDIDataset('K:\\dataset\\e-gmd-v1.0.0\\e-gmd-v1.0.0', sample_length=sample_length, eval=False, OaF_midi=(model_name == 'OaF'))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_data = AudioMIDIDataset('K:\\dataset\\e-gmd-v1.0.0\\e-gmd-v1.0.0', sample_length=sample_length, eval=True, OaF_midi=(model_name == 'OaF')) # ../data/e-gmd-v1.0.0
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    accelerator.wait_for_everyone()

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
        if model_name == 'OaF':
            model = OnsetsAndFrames()
        else:
            model = HTDemucs(**model_args)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        if model_name == 'OaF':
            ckpt_path = os.path.join(out_dir, 'ckpt_OaF.pt')
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint['model']
            model = OnsetsAndFrames()
            unwanted_prefix = 'module.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
        else:
            ckpt_path = os.path.join(out_dir, 'ckpt_HTDemucs.pt')
            checkpoint = torch.load(ckpt_path, map_location=device)
            checkpoint_model_args = checkpoint['model_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['sources', 'audio_channels', 'depth', 't_layers', 't_heads', 't_dropout', 't_weight_decay', 'samplerate', 'segment', 'velocity_branch']:
                model_args[k] = checkpoint_model_args[k]
            # create the model
            model = HTDemucs(**model_args)
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = 'module.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
            torch.cuda.empty_cache()
    elif init_from.startswith('HTDemucs'):
        print(f"Initializing from pretrained weights: {init_from}")
        model = HTDemucs(**model_args)
        # TODO: load the pretrained weights

        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['sources', 'audio_channels', 'depth', 't_layers', 't_heads', 't_dropout', 't_weight_decay', 'samplerate', 'segment', 'velocity_branch']:
            model_args[k] = getattr(model, k)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=False) #(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # loss function
    size = (sample_rate * sample_length, 88) if model_name == 'HTDemucs' else (100 * sample_length, 88)
    #pos_weight = torch.ones(size).to(accelerator.device) * 961958 # empirical class imbalance
    pos_weight = torch.tensor([9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06,
        9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06,
        9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06,
        9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 3.9239e+02, 9.6000e+06,
        9.6000e+06, 9.6000e+06, 1.2916e+03, 9.6000e+06, 9.6000e+06, 9.6000e+06,
        9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06,
        1.4221e+02, 1.8817e+03, 1.0049e+02, 4.1920e+04, 4.5903e+02, 9.6000e+06,
        3.3986e+02, 4.7411e+02, 3.1721e+02, 1.6951e+03, 4.3946e+03, 9.6376e+03,
        5.6685e+02, 3.7353e+04, 4.7491e+03, 3.2195e+02, 6.6243e+03, 2.3800e+03,
        1.8595e+03, 3.2194e+03, 5.9626e+04, 2.9002e+04, 9.3194e+03, 4.1530e+03,
        9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06,
        9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06,
        9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06,
        9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06,
        9.6000e+06, 9.6000e+06, 9.6000e+06, 9.6000e+06]).to(accelerator.device)
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean') #  pos_weight=pos_weight, 
    regression_loss = torch.nn.MSELoss()

    # Processor
    SAMPLE_RATE = 44100
    HOP_LENGTH = 441 # SAMPLE_RATE * 32 // 1000
    N_MELS = 250
    MEL_FMIN = 30
    MEL_FMAX = SAMPLE_RATE // 2
    WINDOW_LENGTH = 2048
    # melspectrogram = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX).to(accelerator.device) # this was the original OaF mel spectrogram class
    melspectrogram = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=WINDOW_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS, center=False, f_min=MEL_FMIN, f_max=MEL_FMAX, pad=(WINDOW_LENGTH-HOP_LENGTH)//2+1).to(accelerator.device)

    # Accelerate
    ctx = nullcontext() #if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype) # ctx = nullcontext() if device_type == 'cpu' else accelerator.autocast()
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

    # compile the model
    # if compile:
    #     print("compiling the model... (takes a ~minute)")
    #     unoptimized_model = model
    #     model = torch.compile(model) # requires PyTorch 2.0

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
            with ctx:
                for batch in loader:
                    print(iters)
                    if iters >= num_iters:
                        break
                    waveform, onset, velocity = batch
                    mel = melspectrogram(waveform) 
                    
                    onset_pred, velocity_pred = model(mel.squeeze(1).transpose(1, 2))
                    
                    #velocity = velocity/torch.amax(velocity, dim=(1,2)).view(-1, 1, 1)
                    #velocity_pred = velocity_pred/torch.amax(velocity_pred, dim=(1,2)).view(-1, 1, 1)

                    loss = bce_loss(onset_pred, onset) + velocity_loss_coeff * regression_loss(velocity_pred, velocity)
                    f1, f1_w_vel = f1_score(onset, onset_pred, velocity, velocity_pred, OaF_midi=(model_name == 'OaF'))
                    if f1 == 0 and f1_w_vel == 0:
                        model.train()
                        print('Divergence detected! Inferencing in training mode.')

                    # f1, f1_w_vel = 0, 0
                    losses[iters] = loss.item()
                    f1_scores[iters] = f1
                    f1_w_vel_scores[iters] = f1_w_vel
                    iters += 1
            
            out[split] = {'Loss': losses.mean().to(accelerator.device), 'F1': f1_scores.mean().to(accelerator.device), 'F1 w/ Vel': f1_w_vel_scores.mean().to(accelerator.device)}
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        lr = 0.98 ** (it//1000) * 1e-4
        return lr

        # # 1) linear warmup for warmup_iters steps
        # if it < warmup_iters:
        #     return learning_rate * it / warmup_iters
        # # 2) if it > lr_decay_iters, return min learning rate
        # if it > lr_decay_iters:
        #     return min_lr
        # # 3) in between, use cosine decay down to min learning rate
        # decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        # assert 0 <= decay_ratio <= 1
        # coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        # return min_lr + coeff * (learning_rate - min_lr)

    # logging
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    accelerator.wait_for_everyone()
    if str(accelerator.device) == 'cuda:1':
        pass
        # time.sleep(10)
    # training loop
    model.train()
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    running_mfu = -1.0
    while True:
        with ctx:
            for batch in train_dataloader:
                with accelerator.accumulate(model):
                    try:
                        audio, onsets, velocities = batch
                        audio, onsets, velocities = shuffled_mixup(batch, num_samples, sample_length)
                        if local_iter_num == 0:
                            print(audio.dtype, onsets.dtype, velocities.dtype)

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
                                        'model': model.state_dict(),
                                        'optimizer': optimizer.state_dict(),
                                        'model_args': model_args,
                                        'iter_num': iter_num,
                                        'best_val_loss': best_val_loss,
                                        'config': config,
                                    }
                                    print(f"saving checkpoint to {out_dir}")
                                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt_' + model_name + '_Transformer_Shuffle_Unweighted_20s_RoPE_step' + str(iter_num) + '.pt'))
                        if iter_num == 0 and eval_only:
                            break

                        accelerator.wait_for_everyone()
                        # forward backward update, with optional gradient accumulation to simulate larger batch size
                        # and using the GradScaler if data type is float16
                        mel = melspectrogram(audio)

                        pred_onsets, pred_vel = model(mel.squeeze(1).transpose(1, 2))

                        #velocities = velocities/torch.amax(velocities, dim=(1,2)).view(-1, 1, 1)
                        #pred_vel = pred_vel/torch.amax(pred_vel, dim=(1,2)).view(-1, 1, 1)

                        loss = bce_loss(pred_onsets, onsets) + velocity_loss_coeff * regression_loss(pred_vel, velocities)

                        # loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

                        # gradient scaling if training in fp16
                        scaler.scale(loss)
                        accelerator.backward(loss)

                        # clip the gradient
                        if grad_clip != 0.0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        # step the optimizer and scaler if training in fp16
                        old_weights = next(model.onset_stack[0].parameters()).clone()
                        scaler.step(optimizer)
                        scaler.update()
                        new_weights = next(model.onset_stack[0].parameters()).clone()

                        # flush the gradients as soon as we can, no need for this memory anymore
                        optimizer.zero_grad(set_to_none=True)

                        # timing and logging
                        t1 = time.time()
                        dt = t1 - t0
                        t0 = t1
                        if iter_num % log_interval == 0 and master_process:
                            # get loss as float. note: this is a CPU-GPU sync point
                            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                            lossf = loss.item()
                            # if local_iter_num >= 5: # let the training loop settle a bit
                            #     mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                            #     running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
                        iter_num += 1
                        local_iter_num += 1
                        # print('iter_num: ', iter_num)
                    except Exception as e:
                        print(e)
                        print('Error occured!')
                        continue
                    # termination conditions
                    if iter_num > max_iters:
                        break
                if iter_num > max_iters:
                    break

if __name__ == '__main__':
    main()