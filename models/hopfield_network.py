import numpy as np


class HopfieldNetwork:
    def __init__(self, num_neurons):
        """
        Inicializa la red de Hopfield.
        num_neurons: Número de neuronas en la red (igual a la longitud del patrón binario).
        """
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        """
        Almacena patrones binarios en la red.
        patterns: Lista de patrones binarios a almacenar (cada patrón debe ser un array 1D de -1 o 1).
        """
        print("Entrenando Red de Hopfield (almacenando patrones)...")
        # Reiniciar pesos
        self.weights = np.zeros((self.num_neurons, self.num_neurons))

        for p in patterns:
            if len(p) != self.num_neurons:
                raise ValueError(f"El patrón debe tener {self.num_neurons} neuronas.")
            p = p.reshape(-1, 1)  # Asegurar que es una columna
            # Regla de Hebbian para actualizar pesos
            self.weights += np.dot(p, p.T)

        # Poner la diagonal en cero (no se auto-conecta)
        np.fill_diagonal(self.weights, 0)
        print("Entrenamiento de Hopfield completado.")

    def recall(self, pattern, max_iter=100):
        """
        Intenta recuperar un patrón almacenado a partir de un patrón de entrada (posiblemente ruidoso).
        pattern: Patrón de entrada (array 1D de -1 o 1).
        """
        if len(pattern) != self.num_neurons:
            raise ValueError(f"El patrón de entrada debe tener {self.num_neurons} neuronas.")

        current_state = np.array(pattern, dtype=float)

        # print(f"Recuperando patrón (estado inicial): {current_state}") # Descomentar para depuración

        for _ in range(max_iter):
            prev_state = np.copy(current_state)

            # Actualización asíncrona de las neuronas
            for i in np.random.permutation(self.num_neurons):
                net_input = np.dot(self.weights[i], current_state)
                current_state[i] = 1 if net_input >= 0 else -1  # Función de activación paso (sign)

            # Si el estado converge, detenemos
            if np.array_equal(current_state, prev_state):
                break

        return current_state

    def compare_patterns(self, pattern1, pattern2):
        """
        Compara dos patrones binarios (-1 y 1) y retorna el número de bits diferentes.
        """
        # Asegúrate de que los patrones sean numpy arrays para la comparación directa
        pattern1 = np.asarray(pattern1)
        pattern2 = np.asarray(pattern2)
        return np.sum(pattern1 != pattern2)