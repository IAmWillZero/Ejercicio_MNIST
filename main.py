import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Cargar el conjunto de datos MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos para mejorar el rendimiento del modelo
x_train = x_train / 255.0
x_test = x_test / 255.0

# Crear el modelo de la red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Aplanar las imágenes de 28x28 a 784 píxeles
    tf.keras.layers.Dense(128, activation='relu'),  # Capa oculta con 128 neuronas y activación ReLU
    tf.keras.layers.Dense(64, activation='relu'),   # Capa oculta con 64 neuronas y activación ReLU
    tf.keras.layers.Dense(10, activation='softmax')  # Capa de salida con 10 neuronas (una por cada dígito)
])

# Compilar el modelo especificando el optimizador y la función de pérdida
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con los datos de entrenamiento
model.fit(x_train, y_train, epochs=5)

# Evaluar el modelo con los datos de prueba y mostrar la precisión
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Precisión del modelo: {accuracy:.4f}')

# Definir el rango de imágenes a predecir
inicio, fin = 0, 15  # Cambia estos valores según lo necesites

# Hacer predicciones y obtener etiquetas reales para el rango especificado
predicciones = model.predict(x_test[inicio:fin])
predicciones_clases = np.argmax(predicciones, axis=1)
etiquetas_reales = y_test[inicio:fin]

# Crear una figura para mostrar las imágenes
plt.figure(figsize=(10, 5))
for i in range(fin - inicio):
    plt.subplot(1, fin - inicio, i + 1)
    plt.imshow(x_test[inicio + i], cmap='gray')
    plt.title(f'Pred: {predicciones_clases[i]}\nReal: {etiquetas_reales[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()