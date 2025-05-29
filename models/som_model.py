from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict


class SOMTrafficLightClassifier:
    def __init__(self, x=10, y=10, input_len=3, sigma=1.0, learning_rate=0.5, random_seed=42):
        """
        Inicializa el SOM.
        x, y: Dimensiones del mapa SOM.
        input_len: Número de características de entrada (ej. 3 para RGB).
        """
        self.som = MiniSom(x, y, input_len, sigma=sigma, learning_rate=learning_rate, random_seed=random_seed)
        self.class_map = {}  # Para mapear neuronas a clases después del entrenamiento

    def train(self, data, num_iteration=10000):
        """
        Entrena el SOM con los datos.
        Data debe ser normalizada (ej. con MinMaxScaler).
        """
        print("Entrenando SOM...")
        self.som.random_weights_init(data)
        self.som.train_random(data, num_iteration)
        print("Entrenamiento SOM completado.")

    def classify_map(self, data, labels, class_names):
        """
        Asigna una etiqueta de clase a cada neurona del SOM basándose en los datos de entrenamiento.
        """
        print("Clasificando neuronas del SOM...")
        self.class_map = {}
        winner_counts = defaultdict(lambda: defaultdict(int))

        for i, f in enumerate(data):
            winner = self.som.winner(f)
            class_label = class_names[labels[i]]
            winner_counts[winner][class_label] += 1

        for winner_coords, class_votes in winner_counts.items():
            most_voted_class = max(class_votes, key=class_votes.get)
            self.class_map[winner_coords] = most_voted_class
        print("Neuronas del SOM clasificadas.")

    def predict(self, features):
        """
        Predice la clase de una nueva observación.
        """
        if not self.class_map:
            # Esto puede ocurrir si el SOM no fue clasificado con datos de entrenamiento
            # O si el patrón de entrada no se mapea a ninguna neurona previamente vista.
            # Para fines del proyecto, un "desconocido" es válido si la confianza es baja.
            return "desconocido"
        winner_coords = self.som.winner(features)
        # Retorna la clase mapeada, o "desconocido" si esa neurona no fue mapeada a una clase
        return self.class_map.get(winner_coords, "desconocido")

    def visualize_map(self, data, labels, class_names):
        """
        Visualiza el mapa SOM con las etiquetas de las neuronas.
        """
        plt.figure(figsize=(10, 10))
        plt.pcolor(self.som.distance_map().T, cmap='bone_r')
        plt.colorbar()

        markers = ['o', 's', 'D']
        colors = ['r', 'y', 'g']

        for i, f in enumerate(data):
            w = self.som.winner(f)
            plt.plot(w[0] + .5 + (np.random.rand() - 0.5) * 0.8,
                     w[1] + .5 + (np.random.rand() - 0.5) * 0.8,
                     markers[labels[i]], markerfacecolor='None',
                     markeredgecolor=colors[labels[i]], markersize=12, markeredgewidth=2)

        # Opcional: Mostrar las etiquetas de las neuronas si están clasificadas
        for (x, y), label in self.class_map.items():
            plt.text(x + 0.5, y + 0.5, label[0].upper(), color='blue', ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

        plt.title('Mapa Autoorganizado de Semáforos (SOM)')
        plt.show()