import torch as t
import torch.nn.functional as F

from tqdm import trange

def shrink_fn(x, thresholds, L):
    return x.sign() * F.relu(x.abs() - (thresholds / L))

def fista_step(z, h1, h2, outputs, dictionary, momentum, thresholds, L):
    off_amount = dictionary @ z.T
    off_amount = off_amount - outputs.T # This is [batch_size, activation_dim]
    inner_right = (1 / L) * (dictionary.T @ off_amount)
    z_new = shrink_fn(z - inner_right.T, thresholds, L)

    return z_new + momentum * (h1 - h2), z_new

def fista_optimize(outputs, dictionary, momentum, thresholds, steps=10_000, device=None):
    if device is None:
        device = t.device('cpu')

    loss_record = []

    L = (dictionary @ dictionary.T).norm()

    z_size = (outputs.size(0), dictionary.size(1))

    z, h1, h2 = t.zeros(z_size, device=device), t.zeros(z_size, device=device), t.zeros(z_size, device=device)

    for i in trange(steps):
        if i % 10 == 0:
            loss_record.append(((outputs - z @ dictionary.T) ** 2).mean().item())
        z_new, h_new = fista_step(z, h1, h2, outputs, dictionary, momentum, thresholds, L)
        h2 = h1
        h1 = h_new
        z = z_new
    
    return z, loss_record