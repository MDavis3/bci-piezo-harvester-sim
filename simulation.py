import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sympy as sp

class HarvestPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)  # Increased for better learning.
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # t and g.

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def physics_loss(model, inputs, r=0.005):  # Vectorized: No loop, batched.
    preds = model(inputs)  # (N,2)
    t = torch.clamp(preds[:,0], min=1e-6)  # (N,)
    g = torch.clamp(preds[:,1], min=0.01)
    P0 = inputs[:,0]  # (N,)
    sigma = (3 * P0 * r**2) / (8 * t**2)  # Corrected standard diaphragm formula.
    V = g * t * sigma
    target_V = 0.5  # Guide to realistic range.
    return torch.mean((V - target_V)**2)  # Scalar loss.

model = HarvestPINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
data_loss_fn = nn.MSELoss()

# Lit-based data: P0 133-666Pa, f1-2Hz, noise0-0.5, temp20-40, inflam0-0.2; t 10-100um, g 0.02-0.05 Vm/N (PZT/PVDF).
num_samples = 1000
inputs = torch.rand(num_samples, 5) * torch.tensor([533,1,0.5,20,0.2]) + torch.tensor([133,1,0,20,0])
targets_t = torch.rand(num_samples) * 9e-5 + 1e-5  # m
targets_g = torch.rand(num_samples) * 0.03 + 0.02  # Vm/N
targets = torch.stack([targets_t, targets_g], dim=1)

for _ in range(1000):  # More epochs.
    preds = model(inputs)
    data_loss = data_loss_fn(preds, targets)
    phys_loss = physics_loss(model, inputs)
    total_loss = data_loss + 0.1 * phys_loss  # Lower weight for stability.
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'pinn_model.pth')  # Save for reuse.

# Constants: Scaled r=0.005m (~2x4mm effective), epsilon for PZT.
epsilon_r = 1700
epsilon_0 = 8.85e-12
r = 0.005

def power_calc(P0, f, noise, temp, inflam, hybrid_factor=1.0):  # Added hybrid multiplier (e.g., 2x for piezo+ultrasound).
    inp = torch.tensor([[float(P0), float(f), float(noise), float(temp), float(inflam)]])
    pred = model(inp)[0]
    t = max(pred[0].item(), 1e-6)
    g_adj = max(pred[1].item(), 0.01)
    A = np.pi * r**2
    sigma = (3 * P0 * r**2) / (8 * t**2)  # Corrected.
    V = g_adj * t * sigma
    C = (epsilon_r * epsilon_0 * A) / t
    temp_factor = 1 - 0.01 * max(temp - 37, 0)
    inflam_factor = 1 - inflam
    P_avg = 0.5 * C * V**2 * f * (1 - noise) * temp_factor * inflam_factor * hybrid_factor
    return P_avg * 1e6, t  # Output in μW.

def monte_carlo_sim(P0_mean=400, f=1.5, noise_mean=0.1, temp_mean=37, inflam_mean=0.1, num_runs=100, hybrid_factor=1.0):
    powers = []
    for _ in range(num_runs):
        P0 = max(133, min(666, np.random.normal(P0_mean, 100)))  # Clamp to lit ranges.
        noise = max(0, min(0.5, np.random.normal(noise_mean, 0.05)))
        temp = max(20, min(40, np.random.normal(temp_mean, 2)))
        inflam = max(0, min(0.2, np.random.normal(inflam_mean, 0.05)))
        P_avg, _ = power_calc(P0, f, noise, temp, inflam, hybrid_factor)
        powers.append(max(0, P_avg))  # Prevent negatives.
    mean_power = np.mean(powers)
    std_power = np.std(powers)
    ci = 1.96 * std_power / np.sqrt(num_runs)
    print(f"Monte Carlo: Mean power {mean_power:.2f} μW, Std {std_power:.2f}, 95% CI ±{ci:.2f}")
    return powers

# Test.
P_avg, opt_t = power_calc(400, 1.5, 0.1, 37, 0.1)
print(f"Optimized thickness: {opt_t*1e3:.2f} mm, Avg power: {P_avg:.2f} μW")

powers_mc = monte_carlo_sim()
plt.hist(powers_mc, bins=20)
plt.xlabel('Power (μW)')
plt.ylabel('Frequency')
plt.title('Monte Carlo Power Distribution')
plt.show()