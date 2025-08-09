import streamlit as st
from simulation import power_calc, monte_carlo_sim, model  # Load model if saved.
import matplotlib.pyplot as plt
import numpy as np
import torch

model.load_state_dict(torch.load('pinn_model.pth'))  # Load saved model.
model.eval()

st.title("BCI Piezo Energy Harvesting Simulator")

P0 = st.slider("Pressure Amplitude (Pa)", 133.0, 666.0, 400.0)
f = st.slider("Frequency (Hz)", 1.0, 2.0, 1.5)
noise = st.slider("Noise Level (0-0.5)", 0.0, 0.5, 0.1)
temp = st.slider("Temperature (°C)", 20.0, 40.0, 37.0)
inflam = st.slider("Inflammation Factor (0-0.2)", 0.0, 0.2, 0.1)
hybrid = st.slider("Hybrid Factor (e.g., 2x for piezo+ultrasound)", 1.0, 5.0, 1.0)

if st.button("Run Single Simulation"):
    P_avg, opt_t = power_calc(P0, f, noise, temp, inflam, hybrid)
    st.write(f"Optimized Thickness: {opt_t*1e3:.2f} mm")
    st.write(f"Average Power: {P_avg:.2f} μW")

if st.button("Run Monte Carlo (100 simulations)"):
    powers_mc = monte_carlo_sim(P0_mean=P0, noise_mean=noise, temp_mean=temp, inflam_mean=inflam, hybrid_factor=hybrid)
    fig, ax = plt.subplots()
    ax.hist(powers_mc, bins=20)
    ax.set_xlabel('Power (μW)')
    ax.set_ylabel('Frequency')
    ax.set_title('Monte Carlo Power Distribution')
    st.pyplot(fig)