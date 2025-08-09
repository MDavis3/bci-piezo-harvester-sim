# BCI Energy Harvester Simulation

Simulates piezoelectric energy harvesting for battery-free brain implants, optimizing for CSF pressure, temperature, inflammation. Uses PINN for physics-accurate predictions, targeting ~1-100 μW outputs.

## Improvements
- Vectorized for speed/no errors.
- Corrected stress equation to standard diaphragm model.
- Scaled for realistic μW yields.
- Added hybrid harvesting and model saving.

## How to Run
1. `pip install torch numpy matplotlib streamlit sympy`
2. `python simulation.py` (trains/saves model, runs test).
3. `streamlit run app.py` for GUI.

## Notes
- Based on 2025 piezo advances (e.g., PZT yields).
- Customize for BCI chips: Adjust r=0.005m.
- Outputs align with low-power needs (~1 mW targets).