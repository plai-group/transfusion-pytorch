import sys
# Add plaicraft-diffusion-model to path (handles editable install issues)
sys.path.insert(0, '/ubc/cs/home/f/fwood/Projects/plaicraft/plaicraft-diffusion-model')

import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from functools import partial
from transfusion_pytorch import Transfusion
from transfusion_pytorch.transfusion import print_modality_sample, divisible_by
import pprint

# plaicraft-diffusion-model is installed with flat structure
from data import PlaicraftMapDatasetFixed

"""
Plaicraft Transfusion Training Script

Trains a Transfusion model on Plaicraft dataset with multiple modalities:
- Audio (in/out): Flow-matched continuous modality
- Video: 3 frames (2 history + 1 prediction), flow-matched
- Keyboard/Mouse: Flow-matched continuous modality

All modalities are jointly trained as conditional flow models (no text/autoregressive component).
"""



# Import Plaicraft dataloaders


# Training configuration
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_TRAIN_STEPS = 100_000
EVAL_EVERY = 1_000
SAMPLE_EVERY = 5_000
CHECKPOINT_EVERY = 10_000
EMA_BETA = 0.9999
CHECKPOINT_PATH = Path('results/plaicraft_checkpoints')
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

# Model configuration
DIM = 128
DEPTH = 2
HEADS = 2
DIM_HEAD = 32

# ============ Configuration ============
dataset_path = "/ubc/cs/research/ubc_ml/plaicraft/data/processed"  # Update this!
global_database_path = "/ubc/cs/research/ubc_ml/plaicraft/data/versioning/global_databases/version_continuous_audio_hdf5/6.6k_6709_players_ids/global_database_training.db"  # Update this!
player_names = ["Dante"]  # Or None to use all players

# Dataset parameters
modalities = ["video", "audio_speak", "audio_hear", "KeyAndMouse"]  # Which modalities to load
window_length_frames = 6 # Number of frames per sample

# Training parameters
batch_size = 1
num_epochs = 5
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f=lambda x: {'tensor':tuple(x.shape)} if hasattr(x,'shape') else \
           {'list':[f(i) for i in x]} if isinstance(x,list) else \
           {'tuple':[f(i) for i in x]} if isinstance(x,tuple) else \
           {'dict':{k:f(v) for k,v in x.items()}} if isinstance(x,dict) else \
           repr(x); 

# Custom collate function to convert Plaicraft batch to Transfusion format
def transfusion_collate_fn(batch):
    """
    Convert Plaicraft batch format to Transfusion multi-modality format.
    
    Expected Plaicraft batch keys:
    - 'audio_speak': (B, C, T) or (B, T, C)
    - 'audio_hear': (B, C, T) or (B, T, C)
    - 'video': (B, T=3, C, H, W)
    - 'keyboard': (B, KB_DIM)
    - 'mouse': (B, MOUSE_DIM)
    
    Transfusion format: List[List[Tuple[int, Tensor]]]
    Each sample is a list of (modality_id, data) tuples.
    """
    # First use plaicraft collate if it exists
    if callable(plaicraft_collate_fn):
        batch = plaicraft_collate_fn(batch)
    
    # Now convert to Transfusion format
    # the output from plaicraft_collate_fn is a dict with format like (here batch B=4, number of "frames"=25, each data frame has its own shape per modality - 
    # key_press is 2x5x16, mouse_movement is 2x10x2, audio in is 15x128, audio_hear is 15x128, video is 2x4x96x160)
    # the fundamental "frame" is 2 units of each modality
    # the following is an example of the shape of the output from plaicraft_collate with B=4, num_frames=25
        # {'dict': {'action': {'dict': {'key_press': {'tensor': (4, 25, 2, 5, 16)},
        #                             'mouse_movement': {'tensor': (4, 25, 2, 10, 2)}}},
        #         'audio_speak': {'tensor': (4, 25, 15, 128)},
        #         'audio_hear': {'tensor': (4, 25, 15, 128)},
        #         'metadata': {'list': [{'list': [{'dict': {'end_frame': '36050',
        #                                                     'player_email': "'c57fa94e436cf49a929d0168e47d26fec3d900b321775e280ef136979c01d5a4'",
        #                                                     'player_gender': "'Male'",
        #                                                     'player_id': '38',
        #                                                     'player_name': "'Dante'",
        #                                                     'player_skill_level': "'Regular'",
        #                                                     'session_id': "'bf7fe22372485157'",
        #                                                     'session_start_timestamp': '1722652425355',
        #                                                     'start_frame': '36000',
        #                                                     'window_length_frames': '50'}}]},
        #                                 {'list': [{'dict': {'end_frame': '8450',
        #                                                     'player_email': "'c57fa94e436cf49a929d0168e47d26fec3d900b321775e280ef136979c01d5a4'",
        #                                                     'player_gender': "'Male'",
        #                                                     'player_id': '38',
        #                                                     'player_name': "'Dante'",
        #                                                     'player_skill_level': "'Regular'",
        #                                                     'session_id': "'4494f7aed47f3315'",
        #                                                     'session_start_timestamp': '1725851918211',
        #                                                     'start_frame': '8400',
        #                                                     'window_length_frames': '50'}}]},
        #                                 {'list': [{'dict': {'end_frame': '4800',
        #                                                     'player_email': "'c57fa94e436cf49a929d0168e47d26fec3d900b321775e280ef136979c01d5a4'",
        #                                                     'player_gender': "'Male'",
        #                                                     'player_id': '38',
        #                                                     'player_name': "'Dante'",
        #                                                     'player_skill_level': "'Regular'",
        #                                                     'session_id': "'10e3b2ff95d6ffcb'",
        #                                                     'session_start_timestamp': '1738724872195',
        #                                                     'start_frame': '4750',
        #                                                     'window_length_frames': '50'}}]},
        #                                 {'list': [{'dict': {'end_frame': '16900',
        #                                                     'player_email': "'c57fa94e436cf49a929d0168e47d26fec3d900b321775e280ef136979c01d5a4'",
        #                                                     'player_gender': "'Male'",
        #                                                     'player_id': '38',
        #                                                     'player_name': "'Dante'",
        #                                                     'player_skill_level': "'Regular'",
        #                                                     'session_id': "'06bb4a9d48db3b88'",
        #                                                     'session_start_timestamp': '1732075203815',
        #                                                     'start_frame': '16850',
        #                                                     'window_length_frames': '50'}}]}]},
        #         'transcript_in': {'list': [{'list': []},
        #                                     {'list': [{'tuple': ["'I'",
        #                                                         '844561',
        #                                                         '844601']},
        #                                             {'tuple': ['"don\'t"',
        #                                                         '844621',
        #                                                         '844741']},
        #                                             {'tuple': ["'have'",
        #                                                         '844761',
        #                                                         '844841']},
        #                                             {'tuple': ["'any'",
        #                                                         '844861',
        #                                                         '844961']}]},
        #                                     {'list': []},
        #                                     {'list': []}]},
        #         'transcript_out': {'list': [{'list': []},
        #                                     {'list': [{'tuple': ["'iron.'",
        #                                                         '840324',
        #                                                         '840645']},
        #                                                 {'tuple': ["'Take'",
        #                                                         '841085',
        #                                                         '841305']},
        #                                                 {'tuple': ["'some,'",
        #                                                         '841365',
        #                                                         '841646']},
        #                                                 {'tuple': ["'like,'",
        #                                                         '841726',
        #                                                         '842026']},
        #                                                 {'tuple': ["'carrots,'",
        #                                                         '842787',
        #                                                         '843347']},
        #                                                 {'tuple': ["'I'",
        #                                                         '843447',
        #                                                         '843507']},
        #                                                 {'tuple': ["'guess.'",
        #                                                         '843547',
        #                                                         '843907']}]},
        #                                     {'list': []},
        #                                     {'list': []}]},
        #         'valid_mask': {'tensor': (4, 25, 2, 1)},
        #         'video': {'tensor': (4, 25, 2, 4, 96, 160)}}}
        # the goal is to convert this into a datastructure compatible with Transfusion where each sample (for Transfusion) 
        # is one "frame" of audio_hear, one "frame" of video, then one "frame" of keyboard, one "frame" of mouse, one "frame" of audio_speak
        # in transfusion speak i believe this should result in each sample having modalities like
        #     modality_default_shape=(
        #     (15,128),          # Audio In: 100 time steps
        #     (15,128),          # Audio Out: 100 time steps
        #     (2, 4, 96, 160),     # Video: 2 frames of four channel at 96x160
        #     (2, 5, 16),              # Keyboard: scalar embedding
        #     (2, 10, 2)               # Mouse: scalar state
        # ),
        # modality_num_dim=(2, 2, 4, 3, 3),  # Dimensionality of each modality
        # with then each sample in transfusion being structured as a list of (modality_id, data) tuples like:
        # [
        #   (0, audio_hear_frame_0),  # modality 0: audio_hear
        #   (1, video_frame_0),      # modality 1: video
        #   (2, keyboard_frame_0),   # modality 2: keyboard
        #   (3, mouse_frame_0),      # modality 3: mouse
        #   (4, audio_speak_frame_0),   # modality 4: audio_speak
        #   (5, audio_hear_frame_1),  # modality 0: audio_hear
        #   (6, video_frame_1),      # modality 1: video
        #   (7, keyboard_frame_1),   # modality 2: keyboard
        #   (8, mouse_frame_1),      # modality 3: mouse
        #   (9, audio_speak_frame_1),   # modality 4: audio_speak
        #   
        # ]
        # note that the video is really a 4 channel 96 x 160 image and should have appropriate positional embeddings applied in transfusion (handled by transfusion itself
        # and that audio is just a 15 x 128 tensor per frame (no positional embeddings needed).  relative time (to each other) positional embeddings should be used for the continuous modalities (audio, keyboard, mouse)

        # ---------------------------------------------------------------------------------
        # Expected input (after Plaicraft's own collate):
        #   batch: Dict with keys
        #     - 'audio_speak':  Tensor[B, T, 15, 128]
        #     - 'audio_hear': Tensor[B, T, 15, 128]
        #     - 'video':     Tensor[B, T, 2, 4, 96, 160]
        #     - 'action':    {
        #           'key_press':      Tensor[B, T, 2, 5, 16],
        #           'mouse_movement': Tensor[B, T, 2, 10, 2]
        #       }
        #     - 'valid_mask': Tensor[B, T, 2, 1]  (optional use to filter invalid pairs)
        #
        # Goal output (Transfusion format):
        #   List over batch of sequences; each sequence is a List[Tuple[int, Tensor]]
        #   For each time index t in [0..T-1], we append the modalities in this order:
        #     0: audio_hear frame (shape [15, 128])
        #     1: video      pair (shape [2, 4, 96, 160])
        #     2: keyboard   pair (shape [2, 5, 16])
        #     3: mouse      pair (shape [2, 10, 2])
        #     4: audio_speak   frame (shape [15, 128])
        #
        # Notes
        # - We treat the "fundamental frame" as a pair for modalities that are naturally paired
        #   in the dataset (video/key_press/mouse_movement -> leading dimension of 2) and as a
        #   single frame for audio modalities (audio_speak/out -> no leading pair dimension).
        # - We keep raw shapes so the Transfusion model can apply axial positional embeddings
        #   for 2D/3D modalities and relative time embeddings for 1D/paired modalities.
        # - If 'valid_mask' is present and both elements in the pair are invalid for a time
        #   index, we skip appending that time step entirely to avoid training on invalid data.
        # ---------------------------------------------------------------------------------

        # Basic validations and shape extraction
        assert isinstance(batch, dict), "Expected a dict from Plaicraft collate_fn"

        audio_speak = batch.get('audio_speak')
        audio_hear = batch.get('audio_hear')
        video = batch.get('video')
        action = batch.get('KeyAndMouse', {})
        key_press = None if action is None else action.get('key_press')
        mouse = None if action is None else action.get('mouse_movement')
        valid_mask = batch.get('valid_mask', None)

        # Ensure required keys are present
        required = {
            'audio_speak': audio_speak,
            'audio_hear': audio_hear,
            'video': video,
        }
        missing = [k for k, v in required.items() if v is None]
        assert not missing, f"Missing required modalities from collate: {missing}"

        B = audio_speak.shape[0]
        T = audio_speak.shape[1]

        # Consistency checks across modalities on B and T
        def _check_bt(tensor, name):
            assert tensor.shape[0] == B and tensor.shape[1] == T, \
                f"{name} must share batch/time dims with audio_speak: got {tuple(tensor.shape[:2])} vs {(B, T)}"

        _check_bt(audio_hear, 'audio_hear')
        _check_bt(video, 'video')
        if key_press is not None:
            _check_bt(key_press, 'action.key_press')
        if mouse is not None:
            _check_bt(mouse, 'action.mouse_movement')
        if valid_mask is not None:
            _check_bt(valid_mask, 'valid_mask')

        # Build Transfusion batch: list over samples
        transfusion_batch = []


        # Causal ordering: input modalities at time t condition output modalities at time t+1
        # Order per timestep: keyboard_{t}, mouse_{t}, audio_speak_{t} -> audio_hear_{t+1}, video_{t+1}
        # This ensures that actions/audio_speak at time t predict the resulting audio_hear and video at t+1
        # We iterate from t=0 to T-2 (using t and t+1 pairs)
        for b in range(B):
            sample_seq = []  # sequence for this sample (list of (modality_id, tensor))
            
            # Add dummy token to avoid empty text list error (will be ignored with ignore_index=-1)
            #sample_seq.extend(torch.tensor([-1], dtype=torch.long, device=audio_speak.device))

            for t in range(1, T - 1):  # Stop at T since we access t+1
                sample_seq.append((3, key_press[b, t-1, 0].float()))   
                sample_seq.append((4, mouse[b, t-1, 0].float())) 
                sample_seq.append((2, video[b, t-1, 0].float())) 
                sample_seq.append((3, key_press[b, t-1, 1].float()))   
                sample_seq.append((4, mouse[b, t-1, 1].float())) 
                sample_seq.append((2, video[b, t-1, 1].float()))
                sample_seq.append((0, audio_speak[b, t-1].float()))
                sample_seq.append((1, audio_hear[b, t-1].float()))
                sample_seq.append((0, audio_speak[b, t].float()))
                sample_seq.append((4, mouse[b, t, 0].float()))
                sample_seq.append((3, key_press[b, t, 0].float()))
                sample_seq.append((4, mouse[b, t, 1].float()))
                sample_seq.append((3, key_press[b, t, 1].float()))
                sample_seq.append((2, video[b, t, 0].float())) 
                sample_seq.append((2, video[b, t, 1].float())) 
                sample_seq.append((1, audio_hear[b, t].float()))


            transfusion_batch.append(sample_seq)
        # print("INSIDE TRANSFUSION COLLATE FN OUTPUT:")
        # pprint.pprint(f(transfusion_batch))
        return transfusion_batch


print(f"Using device: {device}")

# ============ Initialize Dataset ============
print("Initializing dataset...")
dataset = PlaicraftMapDatasetFixed(
    dataset_path=dataset_path,
    modalities=modalities,
    window_length_frames=window_length_frames,
    hop_length_frames=window_length_frames,  # No overlap
    player_names=player_names,
    global_database_path=global_database_path
)

plaicraft_collate_fn=dataset.collate_fn


print(f"Dataset initialized with {len(dataset)} samples")

sample = dataset[0]
print("Sample keys:", sample.keys())




pprint.pprint(f(sample))

# ============ Create DataLoader ============
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=transfusion_collate_fn,
    pin_memory=True if device.type == "cuda" else False
)

# draw a single batch to test
batch = next(iter(dataloader))
print(f"Batch size: {len(batch)}")
print(f"Type of batch: {type(batch)}")
print(f"Type of batch[0]: {type(batch[0])}")
print(f"Type of batch[0][0]: {type(batch[0][0])}")
print(f"batch[0][0] = {batch[0][0]}")
for i, sample in enumerate(batch):
    print(f" Sample {i}:")
    pprint.pprint(f(sample))    


    # def __init__(
    #     self,
    #     *,
    #     num_text_tokens,
    #     transformer: dict | Transformer,
    #     dim_latent: int | tuple[int, ...] | None = None,
    #     channel_first_latent: bool | tuple[bool, ...] = False,
    #     add_pos_emb: bool | tuple[bool, ...] = False,
    #     modality_encoder: Module | tuple[Module, ...] | None = None,
    #     modality_decoder: Module | tuple[Module, ...] | None = None,
    #     pre_post_transformer_enc_dec: tuple[Module, Module] | tuple[tuple[Module, Module], ...] | None = None,
    #     modality_default_shape: tuple[int, ...] | tuple[tuple[int, ...], ...] | None = None,
    #     fallback_to_default_shape_if_invalid = False,
    #     modality_num_dim: int | tuple[int, ...] | None = None,
    #     to_modality_shape_fn: Callable | tuple[Callable, ...] = default_to_modality_shape_fn,
    #     ignore_index = -1,
    #     flow_loss_weight = 1.,
    #     text_loss_weight = 1.,
    #     velocity_consistency_loss_weight = 0.1,
    #     reconstruction_loss_weight = 0.,
    #     modality_encoder_decoder_requires_batch_dim = True, # whether the modality encoder / decoder requires batch dimension, will auto assume it is needed
    #     odeint_kwargs: dict = dict(
    #         atol = 1e-5,
    #         rtol = 1e-5,
    #         method = 'midpoint'
    #     ),
    # ):

# Create Transfusion model with multiple modalities
print('Initializing Transfusion model...')
model = Transfusion(
    num_text_tokens=0,  # No text modality
    transformer = dict(
        dim = DIM,
        depth = DEPTH,
        dim_head = DIM_HEAD,
        heads = HEADS,
        use_flex_attn = True,
    ),
    # Modality configurations matching collate output:
    # Modality 0: audio_speak   - shape [15, 128] - 1D temporal signal
    # Modality 1: audio_hear  - shape [15, 128] - 1D temporal signal  
    # Modality 2: video      - shape [4, 96, 160] - 2D spatial (C, H, W)
    # Modality 3: key_press  - shape [5, 16] - 2D embedding (keys, features)
    # Modality 4: mouse      - shape [10, 2] - 2D trajectory (steps, xy)
    dim_latent=(128, 128, 4, 16, 2),
    
    modality_default_shape=(
        (15,),          # audio_speak: 15 time steps (channel dim separate in dim_latent)
        (15,),          # audio_hear: 15 time steps (channel dim separate in dim_latent)
        (96, 160),      # video: 96 height × 160 width (channel dim separate in dim_latent)
        (5,),           # keyboard: 5 keys (embedding dim separate in dim_latent)
        (10,),          # mouse: 10 time steps (xy coords dim separate in dim_latent)
    ),
    
    modality_num_dim=(1, 1, 2, 1, 1),  # Number of spatial/temporal dimensions per modality
    
    # channel_first_latent: whether input is (C, ...) vs (..., C)
    # - audio: (15, 128) is (time, features) → channel-last → False
    # - video frame(s): (4, 96, 160) is (C, H, W) → channel-first → True  
    # - keyboard/mouse: (keys, features) / (time, xy) → channel-last → False
    channel_first_latent=(False, False, True, False, False),
    
    # add_pos_emb: inject learned positional embeddings
    # - audio: temporal position embeddings helpful
    # - video frame(s): spatial (H, W) axial embeddings
    add_pos_emb=(True, True, True, False, False),
    
    # Training settings
    ignore_index=-1,
    
    # Flow matching settings
    odeint_kwargs=dict(
        method='midpoint',
        atol=1e-5,
        rtol=1e-5
    )
).to(device)

print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

# Create EMA model for stable inference
ema_model = model.create_ema(beta=EMA_BETA)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# Create dataloader
train_loader = dataloader

# Training loop
print('Starting training...')
step = 0
model.train()

while step < NUM_TRAIN_STEPS:
    for batch in train_loader:
        if step >= NUM_TRAIN_STEPS:
            break
        
        # Forward pass
        # fixme: batch is empty here
        loss = model(batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update EMA
        ema_model.update()
        
        # Logging
        if divisible_by(step, 100):
            print(f'Step {step}/{NUM_TRAIN_STEPS} | Loss: {loss.item():.4f} ')
        
        # Evaluation
        # if divisible_by(step, EVAL_EVERY):
        #     model.eval()
        #     with torch.no_grad():
        #         # Sample from EMA model with conditioning from first batch sample
        #         print('\n--- Sampling from EMA model ---')
        #         # Use first few modalities from batch as prompt for conditional generation
        #         if len(batch) > 0 and len(batch[0]) > 0:
        #             # Take first 8 modalities as prompt (audio_speak, video, keyboard, mouse sequence)
        #             prompt = batch[0][:min(8, len(batch[0]))]
        #             sample = ema_model.sample(
        #                 prompt=prompt,
        #                 max_length=512,  # Adjust based on expected total sequence length
        #                 temperature=1.0
        #             )
        #             print_modality_sample(sample)
        #         else:
        #             print("Skipping sampling - empty batch")
        #     model.train()
        
        # Save checkpoint
        if divisible_by(step, CHECKPOINT_EVERY):
            ckpt_path = CHECKPOINT_PATH / f'checkpoint_step_{step}.pt'
            
            # Direct save avoids debugger inspection overhead
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            
            # Log checkpoint info without triggering slow repr
            ckpt_size_mb = ckpt_path.stat().st_size / 1e6 if ckpt_path.exists() else 0
            print(f'✓ Checkpoint saved: {ckpt_path.name} ({ckpt_size_mb:.1f} MB)')
        
        step += 1

print('Training complete!')

# Save final model
final_checkpoint = {
    'step': step,
    'model_state_dict': model.state_dict(),
    'ema_model_state_dict': ema_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(final_checkpoint, CHECKPOINT_PATH / 'final_model.pt')
print(f'Saved final model to {CHECKPOINT_PATH / "final_model.pt"}')