import torch as t

from nnsight import LanguageModel

from buffer import ActivationBuffer
from fista import fista_optimize

from datasets import load_dataset

import matplotlib.pyplot as plt

device_name = 'cuda:0'
device = t.device(device_name)

model_name = 'EleutherAI/pythia-70m-deduped'
model = LanguageModel(model_name)

dataset_name = 'monology/pile-uncopyrighted'
dataset = iter(load_dataset(dataset_name, streaming=True, split='train'))

activation_dim = 512
scale_factor = 32
dictionary_dim=activation_dim * scale_factor

buf = ActivationBuffer(model, dataset, batch_size=128, activation_dim=activation_dim)

batch = next(buf).to(device)

dictionary = t.rand(activation_dim, dictionary_dim).to(device)

value, loss_record = fista_optimize(batch, dictionary, 0.4, 1, device=device)

print(value)
print((value != 0).sum(dim=-1))
print(value.norm(p=1, dim=-1))

print(((batch - value @ dictionary.T) ** 2).mean())

plt.figure()
plt.plot(loss_record)
plt.savefig("loss_record.png")