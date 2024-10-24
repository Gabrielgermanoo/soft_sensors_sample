import numpy as np
import matplotlib.pyplot as plt

# Constants
V_input = 5.0  # Applied voltage in volts
R_initial = 1000  # Initial resistance in ohms (without pressure)
pressure_coefficient = 0.02  # Resistance change per unit of pressure

# Simulated pressure data (in arbitrary units)
pressure_values = np.linspace(0, 100, 100)  # Pressure from 0 to 100 units

# Calculate the resistance change due to pressure
resistance_values = R_initial * (1 + pressure_coefficient * pressure_values)

# Calculate the current using Ohm's Law: I = V / R
current_values = V_input / resistance_values

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(pressure_values, resistance_values, label="Resistance")
plt.xlabel("Pressure (arbitrary units)")
plt.ylabel("Resistance (Ohms)")
plt.title("Resistance vs. Applied Pressure")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(pressure_values, current_values, label="Current", color='orange')
plt.xlabel("Pressure (arbitrary units)")
plt.ylabel("Current (Amps)")
plt.title("Current vs. Applied Pressure")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
