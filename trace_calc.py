#!/usr/bin/env python3
"""10000-step trace calculation for hub neuron."""

tau_hub = 4.39997
decay_hub = 1.0 / tau_hub
h1_spk = 1.0

# 1. Exact formula from kmnist_solver.py (line 136):
#    hub1 = hub1 + decay_hub * (-hub1 + h1_spk)
hub_in_code = 0.0
for _ in range(10000):
    hub_in_code = hub_in_code + decay_hub * (-hub_in_code + h1_spk)

# 2. Formula if spike is added WITHOUT multiplying by decay_hub:
#    hub1 = hub1 - decay_hub * hub1 + h1_spk
hub_unscaled = 0.0
for _ in range(10000):
    hub_unscaled = hub_unscaled - decay_hub * hub_unscaled + h1_spk

print(f"tau_hub = {tau_hub}, decay_hub = 1.0 / {tau_hub} = {decay_hub}")
print(f"After 10000 steps:")
print(f"1) Exact line 136: hub = hub + decay_hub * (-hub + 1.0)  --->  {hub_in_code}")
print(f"2) Unscaled spike: hub = hub - decay_hub * hub + 1.0     --->  {hub_unscaled}")
