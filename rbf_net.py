import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Генерация данных
x = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)  # Создаем 1000 значений от 0 до 2π

# Выбираем функцию, которую будем аппроксимировать
# В данном случае это x^2 * sin(x)
y = x**2 * np.sin(x)


# Определение радиально-базисной нейронной сети (RBF-сети)
class RBFNet(tf.keras.Model):
    def __init__(self, num_centers, sigma=0.3):
        super(RBFNet, self).__init__()
        self.num_centers = num_centers  # Количество центров RBF-функций
        self.sigma = sigma  # Ширина гауссовых функций

        # Инициализация центров RBF-функций случайными значениями в диапазоне [0, 2π]
        self.centers = tf.Variable(tf.random.uniform((num_centers, 1), minval=0, maxval=2 * np.pi), trainable=True)

        # Линейный слой (выходной слой) с 1 нейроном, без активации
        self.linear = tf.keras.layers.Dense(1)

    def rbf(self, x):
        """
        Вычисление радиально-базисных функций (RBF) для входного x.
        Используем Гауссову функцию: exp(- (x - center)^2 / (2 * sigma^2))
        """
        return tf.exp(-tf.square(x - tf.transpose(self.centers)) / (2 * self.sigma**2))

    def call(self, x):
        """
        Прямой проход (forward pass):
        1. Вычисляем выход RBF-функций
        2. Передаем его в линейный слой
        """
        rbf_output = self.rbf(x)
        return self.linear(rbf_output)


# Параметры модели
num_centers = 25  # Количество центров RBF-функций
sigma_value = 0.7  # Ширина RBF-функций
learning_rate = 0.01  # Скорость обучения
epochs = 3000  # Количество эпох обучения

# Создание модели
model = RBFNet(num_centers=num_centers, sigma=sigma_value)

# Функция потерь (MSE - среднеквадратичная ошибка)
criterion = tf.keras.losses.MeanSquaredError()

# Оптимизатор Adam
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Обучение модели
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        outputs = model(x)  # Прямой проход (предсказания)
        loss = criterion(y, outputs)  # Вычисление ошибки

    # Вычисление градиентов и обновление весов
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Вывод информации каждые 100 эпох
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.numpy():.4f}')

# Получаем предсказания модели
predicted = model(x).numpy()

# График истинной функции и предсказаний модели
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Истинная функция x^2 * sin(x)', linewidth=2)
plt.plot(x, predicted, label='Предсказание RBF-сети', linestyle='dashed', linewidth=2)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Аппроксимация x^2 * sin(x) с помощью RBF-сети')
plt.grid()
plt.show()

# График активаций радиально-базисных функций
rbf_output = model.rbf(x).numpy()
plt.figure(figsize=(10, 6))
for i in range(num_centers):
    plt.plot(x, rbf_output[:, i], label=f'Центр {i+1}')
plt.title('Активации RBF-функций')
plt.xlabel('x')
plt.ylabel('Активация')
plt.legend()
plt.grid()
plt.show()