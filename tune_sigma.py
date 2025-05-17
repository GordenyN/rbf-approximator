import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Функция, которую аппроксимируем
func = lambda x: x**2 * np.sin(x)
x = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y = func(x)

# Класс модели (тот же самый RBFNet)
class RBFNet(tf.keras.Model):
    def __init__(self, num_centers, sigma):
        super().__init__()
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = tf.Variable(tf.random.uniform((num_centers, 1), minval=0, maxval=2*np.pi), trainable=True)
        self.linear = tf.keras.layers.Dense(1)

    def rbf(self, x):
        return tf.exp(-tf.square(x - tf.transpose(self.centers)) / (2 * self.sigma ** 2))

    def call(self, x):
        rbf_out = self.rbf(x)
        return self.linear(rbf_out)

# Параметры
sigmas_to_try = np.linspace(0.1, 2.0, 20)  # Пробуем 20 значений sigma
num_centers = 25
epochs = 3000
learning_rate = 0.01

losses = []

# Обучение модели для разных sigma
for sigma in sigmas_to_try:
    model = RBFNet(num_centers=num_centers, sigma=sigma)
    criterion = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            pred = model(x)
            loss = criterion(y, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    losses.append(loss.numpy())
    print(f"sigma = {sigma:.3f}, Loss = {loss.numpy():.4f}")

# Находим лучшую sigma
best_idx = np.argmin(losses)
best_sigma = sigmas_to_try[best_idx]
print(f"\n✅ Лучшая sigma: {best_sigma:.3f} с Loss = {losses[best_idx]:.4f}")

# График зависимости loss от sigma
plt.figure(figsize=(8, 5))
plt.plot(sigmas_to_try, losses, marker='o')
plt.xlabel('Sigma')
plt.ylabel('Loss (MSE)')
plt.title('Поиск оптимального sigma')
plt.grid(True)
plt.show()
