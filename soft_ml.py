import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Simulação de dados para um sensor resistivo
np.random.seed(42)
num_samples = 200
resistance_values = np.linspace(900, 1100, num_samples)  # Resistência variando de 900 a 1100 ohms

# Força aplicada simulada (em Newtons), com uma relação não linear com a resistência e algum ruído
force_values = 50 + 0.1 * (1100 - resistance_values)**2 + np.random.normal(0, 5, num_samples)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(resistance_values.reshape(-1, 1), force_values, test_size=0.3, random_state=42)

# Treinamento do modelo de Regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)

# Predição com o conjunto de teste
y_pred = model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')

# Visualização dos dados
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, label="Dados reais", color='blue')
plt.plot(X_test, y_pred, label="Predição do modelo", color='red')
plt.xlabel("Resistência (Ohms)")
plt.ylabel("Força (N)")
plt.title("Predição de Força com Soft Sensor e ML")
plt.legend()
plt.grid()
plt.show()