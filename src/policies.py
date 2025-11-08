import numpy as np
import torch

def epsilon_greedy(q_values: torch.Tensor, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(q_values.shape[-1])
    return int(torch.argmax(q_values, dim=-1).item())

def softmax_policy(q_values: torch.Tensor, temperature: float) -> int:
    # numerically stable softmax over actions
    q = q_values.detach().cpu().numpy().astype(np.float64)
    q = q - q.max()
    probs = np.exp(q / max(temperature, 1e-6))
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
