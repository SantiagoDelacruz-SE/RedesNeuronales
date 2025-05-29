from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression  # O Ridge, etc. para la capa de salida
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial.distance import cdist  # Para calcular distancias euclidianas


class RBFNetwork:
    def __init__(self, n_rbf_neurons=10, gamma=1.0):
        """
        Inicializa la red RBF.
        n_rbf_neurons: Número de neuronas en la capa oculta RBF.
        gamma: Parámetro de ancho para la función Gaussiana.
        """
        self.n_rbf_neurons = n_rbf_neurons
        self.gamma = gamma
        self.centers = None
        self.logistic_regression = None  # Para la capa de salida

    def _calculate_rbf_activations(self, X):
        """
        Calcula las activaciones de la capa RBF.
        """
        if self.centers is None:
            raise ValueError("Los centros RBF no han sido inicializados. Entrena la red primero.")
        # Calcular la distancia euclidiana entre cada punto de X y cada centro
        distances = cdist(X, self.centers, 'euclidean')
        # Aplicar la función de base radial Gaussiana
        return np.exp(-self.gamma * distances ** 2)

    def train(self, X_train, y_train):
        """
        Entrena la red RBF en dos fases:
        1. K-Means para encontrar los centros RBF.
        2. Regresión logística (o similar) para la capa de salida.
        """
        print("Entrenando Red RBF...")
        # Fase 1: Encontrar los centros RBF usando K-Means
        self.kmeans = KMeans(n_clusters=self.n_rbf_neurons, random_state=42, n_init=10)  # n_init para evitar warning
        self.kmeans.fit(X_train)
        self.centers = self.kmeans.cluster_centers_

        # Fase 2: Entrenar la capa de salida
        # Calcular activaciones RBF para los datos de entrenamiento
        rbf_activations_train = self._calculate_rbf_activations(X_train)

        # Usar Regresión Logística como la capa de salida para clasificación
        self.logistic_regression = LogisticRegression(max_iter=1000, random_state=42)
        self.logistic_regression.fit(rbf_activations_train, y_train)
        print("Entrenamiento Red RBF completado.")

    def predict(self, X_test):
        """
        Realiza predicciones con la red RBF entrenada.
        """
        rbf_activations_test = self._calculate_rbf_activations(X_test)
        return self.logistic_regression.predict(rbf_activations_test)

    def evaluate(self, X_test, y_test, class_names):
        """
        Evalúa el rendimiento del modelo RBF.
        """
        y_pred = self.predict(X_test)
        print("\n--- Evaluación de la Red RBF ---")
        print("Reporte de Clasificación:\n", classification_report(y_test, y_pred, target_names=class_names))
        print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))