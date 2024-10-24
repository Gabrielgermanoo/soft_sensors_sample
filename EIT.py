import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Simulate EIT Data (Impedance changes with touch)
np.random.seed(42)
num_samples = 1000
num_electrodes = 16

# Simulate forces applied (in Newtons)
forces = np.random.uniform(0, 100, num_samples)

# Simulate impedance values for each electrode, varying with applied force
# Impedance decreases as force increases (arbitrary relation for simulation)
impedance_data = np.array([1000 - 5 * f + np.random.normal(0, 50, num_electrodes) for f in forces])

# Step 2: Pre-process the data
scaler = StandardScaler()
impedance_data_normalized = scaler.fit_transform(impedance_data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(impedance_data_normalized, forces, test_size=0.2, random_state=42)

# Step 3: Train a Neural Network
model = Sequential()
model.add(Dense(32, input_dim=num_electrodes, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Step 4: Predict and Evaluate the Model
y_pred = model.predict(X_test)

# Plotting actual vs predicted forces
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([0, 100], [0, 100], 'r--', label='Ideal Prediction')
plt.xlabel('Actual Force (N)')
plt.ylabel('Predicted Force (N)')
plt.title('Actual vs Predicted Force using EIT-based Soft Sensor')
plt.legend()
plt.grid()
plt.show()

# Display model performance
mse = np.mean((y_test - y_pred.flatten())**2)
mae = np.mean(np.abs(y_test - y_pred.flatten()))
print(f'Mean Squared Error: {mse:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')
