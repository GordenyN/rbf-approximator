import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Генерация данных (синусойда )
x = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y = np.sin(x)

#Модель
class SineNet(tf.keras.Model):
    def __init__(self):
        super(SineNet, self).__init__()
        self.fc1 = tf.keras.layers.Dense(32, activation='relu')  # Первый слой: 32 нейрона, ReLU
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')  # Второй слой: 64 нейрона, ReLU
        self.fc3 = tf.keras.layers.Dense(1)  # Третий слой: 1 нейрон, без активации

#vtnjl описывает, как данные проходят через слои.
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# инициализация модели
model = SineNet()
criterion = tf.keras.losses.MeanSquaredError()#инициализация функции потерь
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)#инициализация оптимизатора learning_rate - скорость обучения(можно менять)

# Обучение
epochs = 500  # Количество эпох
for epoch in range(epochs):
    with tf.GradientTape() as tape:   # используется для автоматического подсчета градиентов
        outputs = model(x)  # Прямой проход (получаем предсказания)
        loss = criterion(y, outputs)  # Вычисление ошибки
    gradients = tape.gradient(loss, model.trainable_variables)  # Вычисление градиентов
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Обновление весов модели

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.numpy():.4f}')

# Прямой проход и вывод результатов
predicted = model(x).numpy()

plt.plot(x, y, label='Sin')  # Истинный график sin(x)
plt.plot(x, predicted, label='Predict Sin')  # Предсказанный график
plt.legend()
plt.show()