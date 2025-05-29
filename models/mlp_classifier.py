from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class MLPTrafficLightClassifier:
    def __init__(self, hidden_layer_sizes=(100,), max_iter=500, random_state=42):
        """
        Inicializa el clasificador MLP.
        """
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                   max_iter=max_iter,
                                   activation='relu', # Función de activación ReLU
                                   solver='adam',     # Optimizador Adam
                                   random_state=random_state,
                                   verbose=True)      # Para ver el progreso del entrenamiento

    def train(self, X_train, y_train):
        """
        Entrena el modelo MLP.
        X_train: Características de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        """
        print("Entrenando MLP...")
        self.model.fit(X_train, y_train)
        print("Entrenamiento MLP completado.")

    def predict(self, X_test):
        """
        Realiza predicciones con el modelo entrenado.
        X_test: Características para predecir.
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test, class_names):
        """
        Evalúa el rendimiento del modelo e imprime un reporte.
        """
        y_pred = self.predict(X_test)
        print("\n--- Evaluación del MLP ---")
        print("Reporte de Clasificación:\n", classification_report(y_test, y_pred, target_names=class_names))
        print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))