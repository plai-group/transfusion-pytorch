import torch
from torch import randint, randn, zeros
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from transfusion_pytorch import Transfusion, print_modality_sample

def divisible_by(num, den):
    return (num % den) == 0

model = Transfusion(
    dim_latent = (16, 4, 10),
    modality_default_shape = ((2,), (8,8), (4,)),
    transformer = dict(
        dim = 64,
        depth = 1,
        dim_head = 8,
        heads = 2
    ),
    channel_first_latent = (False, True, False),
    modality_num_dim = (1, 2, 1),
    num_text_tokens = 0,
    fallback_to_default_shape_if_invalid = True
).cuda()

class MockDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        # Return two modalities, no text
        return [(0, torch.ones(2, 16)),
                (1, torch.randn(4, 8,8)),
                (2, torch.zeros(4,10))]

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

def collate_fn(data):
    data = [*map(list, data)]
    return data

mock_dataset = MockDataset()

dataloader = DataLoader(mock_dataset, batch_size = 6, collate_fn = collate_fn)
iter_dl = cycle(dataloader)

optimizer = Adam(model.parameters(), lr = 3e-4)

# train loop

for step in range(1, 1000 + 1):
    loss = model(next(iter_dl))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'{step}: {loss.item():.3f}')

print("Training done. Attempting sample...")
try:
    # Sample without prompt (THIS IS WHAT WE WANT TO SUPPORT)
    print("Sampling with no prompt...")
    sample = model.sample(max_length=20)
    print("Sampled successfully!")
    print_modality_sample(sample)

    # Sample with modality prompt (THIS IS ALSO WHAT WE WANT TO SUPPORT)
    print("\nSampling with modality prompt...")
    prompt_modality = (0, torch.ones(2, 16))
    sample_prompt = model.sample(prompt=prompt_modality, max_length=20)
    print("Sampled with prompt successfully!")
    print_modality_sample(sample_prompt)

    # Sample with multi-modality prompt (THIS IS ALSO WHAT WE WANT TO SUPPORT)
    print("\nSampling with multi-modality prompt...")
    prompt_modality = [(0, torch.ones(2, 16)),
                       (1, torch.randn(4, 8,8))]
    sample_prompt = model.sample(prompt=prompt_modality, max_length=20)
    print("Sampled with prompt successfully!")
    print_modality_sample(sample_prompt)


except Exception as e:
    print(f"Sampling failed: {e}")
    import traceback
    traceback.print_exc()
