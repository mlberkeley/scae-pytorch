import torch
from attention import SetTransformer
from capsule import CapsuleLayer

# originally (32, 11, 2)
INPUT = torch.ones((32, 11, 2))

st = SetTransformer(2)
encoded = st.forward(INPUT)

print(encoded.shape)

decoder = CapsuleLayer(input_dims=32, n_caps=3, n_caps_dims=2, n_votes=4, n_caps_params=32, n_hiddens=128,
                       learn_vote_scale=True, deformations=True, noise_type='uniform', noise_scale=4., similarity_transform=False)

decoder.forward(encoded)
