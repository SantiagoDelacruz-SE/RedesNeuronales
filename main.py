from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import cv2  # Necesario para cv2.destroyAllWindows al final
import matplotlib.pyplot as plt  # Importar para que la visualización del SOM funcione

# Importar funciones y clases desde tus módulos
from utils.image_processor import (
    create_dummy_dataset,
    load_real_dataset,
    start_camera_feed,
    extract_color_features,
    load_and_preprocess_image,
    process_raw_images_to_processed  # <--- ¡Nueva función importada!
)
from models.mlp_classifier import MLPTrafficLightClassifier
from models.som_model import SOMTrafficLightClassifier
from models.rbf_classifier import RBFNetwork
from models.hopfield_network import HopfieldNetwork

# Diccionario para mapear índices de clase a etiquetas de texto
CLASS_LABELS = {0: "red", 1: "yellow", 2: "green"}  # Asegúrate que estos coincidan con tus nombres de carpeta
# Diccionario para mapear etiquetas de texto a la señal final
SIGNAL_MAP = {
    "red": "ADVERTENCIA (ALTO)",
    "yellow": "PELIGRO (PRECAUCION)",
    "green": "SEGURIDAD (PUEDE CRUZAR)",
    "desconocido": "Estado desconocido"
}


def get_signal_message(predicted_label):
    if predicted_label in SIGNAL_MAP:
        return SIGNAL_MAP[predicted_label]
    # Si predicted_label es un índice (como en MLP/RBF), mapearlo primero a nombre
    elif isinstance(predicted_label, int) and predicted_label in CLASS_LABELS:
        text_label = CLASS_LABELS[predicted_label]
        return SIGNAL_MAP.get(text_label, "ESTADO DESCONOCIDO")
    else:
        return SIGNAL_MAP["desconocido"]


def run_mlp_prediction(model, class_names, scaler=None):

    class_index_to_name = {i: name for i, name in enumerate(class_names)}

    def predict_frame(frame_features):
        if scaler:  # Si hay un scaler, se asume que las características necesitan escalarse
            frame_features = scaler.transform(frame_features)

        predicted_class_index = model.predict(frame_features)[0]  # Asumiendo que predict devuelve un array de índices
        predicted_label_name = class_index_to_name[predicted_class_index]
        return get_signal_message(predicted_label_name)

    return predict_frame


def run_som_prediction(model, class_names, scaler=None):
    """
    Crea una función de callback para predicciones con el modelo SOM.
    """

    def predict_frame(frame_features):
        if scaler:
            # SOM.predict espera un array 1D. frame_features viene como (1, n_features)
            # Primero escala y luego toma el primer (y único) elemento del batch.
            frame_features_scaled = scaler.transform(frame_features)[0]
        else:
            frame_features_scaled = frame_features[0]  # Toma el primer (y único) elemento del batch

        # El predict del SOM devuelve directamente el nombre de la clase ("red", "yellow", "green", "desconocido")
        predicted_label = model.predict(frame_features_scaled)
        return get_signal_message(predicted_label)

    return predict_frame


def run_rbf_prediction(model, class_names, scaler=None):
    """
    Crea una función de callback para predicciones con el modelo RBF.
    """
    # Invertir el mapeo de class_names para obtener la etiqueta de texto a partir del índice numérico
    class_index_to_name = {i: name for i, name in enumerate(class_names)}

    def predict_frame(frame_features):
        if scaler:
            frame_features = scaler.transform(frame_features)

        predicted_class_index = model.predict(frame_features)[0]  # Asumiendo que predict devuelve un array de índices
        predicted_label_name = class_index_to_name[predicted_class_index]
        return get_signal_message(predicted_label_name)

    return predict_frame


def run_hopfield_prediction(model, stored_patterns_dict):
    """
    Crea una función de callback para predicciones con el modelo Hopfield.
    """

    def binarize_for_hopfield(features):
        # ESTA ES LA FUNCIÓN CRÍTICA PARA HOPFIELD.
        # Necesitas una estrategia ROBUSTA para convertir las características del semáforo
        # (ej. promedios RGB de la ROI) en un patrón binario (-1 o 1)
        # que tenga sentido para la red de Hopfield.

        # Tus características 'features' son un array 2D: [[R, G, B]]
        r, g, b = features[0]  # Desempaquetar los promedios RGB

        # Definir un patrón de 9 elementos para Hopfield (como los almacenados)
        # Inicialmente todos a -1
        binary_pattern = np.full((3, 3), -1, dtype=int).flatten()

        # Lógica de binarización basada en colores dominantes
        # Ajusta estos umbrales para que se adapten a la salida de `extract_color_features`
        # y a los patrones que has definido para Hopfield.

        # Ejemplo de binarización (muy simple y puede necesitar ajuste):
        # Si el rojo es el más dominante y alto, activar el bit central.
        if r > g and r > b and r > 0.6:
            binary_pattern[4] = 1  # Bit central (simulando patrón rojo)
        # Si el verde es el más dominante y alto, activar el bit inferior central.
        elif g > r and g > b and g > 0.6:
            binary_pattern[7] = 1  # Bit inferior central (simulando patrón verde)
        # Si es amarillo (mezcla de rojo y verde, bajo azul)
        elif r > 0.5 and g > 0.5 and b < 0.3 and (abs(r - g) < 0.2):
            binary_pattern[1] = 1  # Bit superior central (simulando patrón amarillo)

        return binary_pattern

    def classify_hopfield_output(recalled_pattern, stored_patterns_dict):
        min_diff = float('inf')
        best_match_label = "desconocido"

        for stored_p_tuple, label in stored_patterns_dict.items():
            stored_p_array = np.array(stored_p_tuple)
            diff = model.compare_patterns(recalled_pattern, stored_p_array)

            # Puedes establecer un umbral de "similaridad" para aceptar una predicción
            # Por ejemplo, si la diferencia es más de 2 bits, considerarlo desconocido
            if diff < min_diff:  # and diff <= 2: # Considera añadir un umbral de diferencia aquí
                min_diff = diff
                best_match_label = label

        # Opcional: Si el mejor match sigue siendo muy diferente, clasificar como desconocido
        # if min_diff > 2: # Ejemplo de umbral de diferencia
        #     return "desconocido", min_diff

        return best_match_label, min_diff

    def predict_frame(frame_features):
        # Primero, binarizar las características del frame para la red de Hopfield
        binary_input_pattern = binarize_for_hopfield(frame_features)

        # Intentar recuperar el patrón
        recalled_pattern = model.recall(binary_input_pattern, max_iter=50)

        # Clasificar el patrón recuperado con respecto a los patrones almacenados
        predicted_label, _ = classify_hopfield_output(recalled_pattern, stored_patterns_dict)
        return get_signal_message(predicted_label)

    return predict_frame


def main():
    print("Iniciando el sistema de observación de semáforos...")

    # --- 1. Preparación de Datos ---
    raw_data_directory = "data/raw"
    processed_data_directory = "data/processed"
    target_img_size = (64, 64)  # Asegúrate de que esto coincida con lo que esperan tus modelos

    # Ejecutar la función para procesar las imágenes RAW y clasificarlas en 'processed'
    print("\n##### PREPROCESANDO IMÁGENES RAW A PROCESSED #####")
    # Este paso intentará clasificar tus imágenes de 'raw' automáticamente por color
    # y guardarlas en 'processed/red', 'processed/yellow', 'processed/green'.
    # ¡Asegúrate de que 'data/raw/' contenga tus imágenes de semáforo!
    process_raw_images_to_processed(raw_data_directory, processed_data_directory, target_img_size)

    # Ahora, cargar el dataset real desde la carpeta 'processed'
    print("\n##### CARGANDO DATASET PARA ENTRENAMIENTO DESDE 'PROCESSED' #####")
    class_labels_list = ["red", "yellow", "green"]  # Nombres de las carpetas/clases esperadas en 'processed'
    X, y, class_mapping = load_real_dataset(processed_data_directory, class_labels_list, target_img_size)

    # Si no hay datos cargados después del preprocesamiento, salir
    if len(X) == 0:
        print("Error: No se encontraron datos de semáforos procesados para cargar.")
        print("Asegúrate de que 'data/raw/' contenga imágenes válidas y que la lógica de clasificación")
        print("en 'process_raw_images_to_processed' sea efectiva para tus imágenes.")
        return

    # Usar los nombres de las clases obtenidos del mapeo real (ej. "red", "yellow", "green")
    # CLASS_LABELS Global se usa para el mapeo a mensajes, pero aquí es la lista de nombres.
    class_names = list(class_mapping.keys())

    # Dividir datos en entrenamiento y prueba para MLP y RBF
    # Para SOM, se suele entrenar con todo el dataset y luego mapear las neuronas.
    # Para Hopfield, se entrenan los patrones discretos.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"\nDatos preparados. Clases: {class_names}")
    print(f"Tamaño del dataset de entrenamiento: {len(X_train)} muestras")
    print(f"Tamaño del dataset de prueba: {len(X_test)} muestras")

    # --- 2. Entrenamiento de Modelos ---

    # MLP
    print("\n##### ENTRENANDO MLP #####")
    # Para MLP, el StandardScaler suele funcionar mejor con las características de color.
    scaler_mlp = StandardScaler()
    X_train_scaled_mlp = scaler_mlp.fit_transform(X_train)
    X_test_scaled_mlp = scaler_mlp.transform(X_test)

    # n_inputs debe ser la cantidad de características (3 para R, G, B promedio)
    # n_outputs debe ser la cantidad de clases (3 para rojo, amarillo, verde)
    mlp_classifier = MLPTrafficLightClassifier(hidden_layer_sizes=(50, 25), max_iter=1000)
    mlp_classifier.train(X_train_scaled_mlp, y_train)
    mlp_classifier.evaluate(X_test_scaled_mlp, y_test, class_names)  # Usar X_test_scaled_mlp para evaluar

    # SOM
    print("\n##### ENTRENANDO SOM #####")
    # SOM usa MinMaxScaler para normalizar los datos a un rango [0, 1]
    # scaler_som = MinMaxScaler()
    # X_scaled_som_train = scaler_som.fit_transform(X_train)
    # X_scaled_som_all = scaler_som.transform(X)  # Para el mapeo de neuronas

    # som_model = SOMTrafficLightClassifier(x=10, y=10, input_len=X_scaled_som_train.shape[1], sigma=0.5,
               #                           learning_rate=0.5)
    # som_model.train(X_scaled_som_train, num_iteration=20000)
    # som_model.classify_map(X_scaled_som_all, y, class_names)
    # Visualizar el mapa SOM (descomentar si quieres ver la ventana emergente)
    # som_model.visualize_map(X_scaled_som_all, y, class_names)

    # RBF
    print("\n##### ENTRENANDO RED RBF #####")
    # RBF también suele beneficiarse de StandardScaler
    scaler_rbf = StandardScaler()
    X_scaled_rbf_train = scaler_rbf.fit_transform(X_train)
    X_scaled_rbf_test = scaler_rbf.transform(X_test)
    rbf_net = RBFNetwork(n_rbf_neurons=10, gamma=0.5)
    rbf_net.train(X_scaled_rbf_train, y_train)
    rbf_net.evaluate(X_scaled_rbf_test, y_test, class_names)

    # Hopfield
    print("\n##### ENTRENANDO RED HOPFIELD #####")
    # Definir patrones binarios simplificados para semáforos (ej. 3x3)
    # Convertir a -1 y 1 para Hopfield
    # Estos patrones deben ser representaciones simbólicas de tus clases.
    # El tamaño del patrón (9 elementos aquí) debe coincidir con la salida de binarize_for_hopfield
    pattern_red = np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]]).flatten()
    pattern_yellow = np.array([[-1, 1, -1], [-1, -1, -1], [-1, -1, -1]]).flatten()
    pattern_green = np.array([[-1, -1, -1], [-1, -1, -1], [-1, 1, -1]]).flatten()

    stored_hopfield_patterns = [pattern_red, pattern_yellow, pattern_green]
    # Mapeo de los patrones almacenados a sus etiquetas de texto.
    hopfield_pattern_labels = {tuple(pattern_red): "red",
                               tuple(pattern_yellow): "yellow",
                               tuple(pattern_green): "green"}

    hopfield_net = HopfieldNetwork(num_neurons=len(pattern_red))
    hopfield_net.train(stored_hopfield_patterns)

    # --- 3. Selección del Modelo a Usar en Tiempo Real y Captura de Cámara ---
    print("\n##### SELECCIÓN DEL MODELO PARA LA CÁMARA #####")

    # --- USAR MLP EN LA CÁMARA ---
    print("\n¡Iniciando cámara con predicciones de MLP! Presiona ESC para salir.")
    camera_prediction_func = run_mlp_prediction(mlp_classifier.model, class_names, scaler=scaler_mlp)
    start_camera_feed(prediction_callback=camera_prediction_func, camera_index=0)

    # --- USAR SOM EN LA CÁMARA ---
    # print("\n¡Iniciando cámara con predicciones de SOM! Presiona ESC para salir.")
    # camera_prediction_func = run_som_prediction(som_model, class_names, scaler=scaler_som)
    # start_camera_feed(prediction_callback=camera_prediction_func, camera_index=0)

    # --- USAR RBF EN LA CÁMARA ---
    # print("\n¡Iniciando cámara con predicciones de RBF! Presiona ESC para salir.")
    # camera_prediction_func = run_rbf_prediction(rbf_net, class_names, scaler=scaler_rbf)
    # start_camera_feed(prediction_callback=camera_prediction_func, camera_index=0)

    # --- USAR HOPFIELD EN LA CÁMARA ---
    # ADVERTENCIA: La binarización para Hopfield en tiempo real es compleja y este ejemplo es MUY básico.
    # Si vas a usar Hopfield, necesitas una forma MUY confiable de convertir la ROI del semáforo
    # en uno de tus patrones binarios almacenados.
    # print("\n¡Iniciando cámara con predicciones de Hopfield! Presiona ESC para salir.")
    # camera_prediction_func = run_hopfield_prediction(hopfield_net, hopfield_pattern_labels)
    # start_camera_feed(prediction_callback=camera_prediction_func, camera_index=0)

    print("\nPrograma finalizado.")
    cv2.destroyAllWindows()  # Asegurarse de cerrar todas las ventanas de OpenCV


if __name__ == "__main__":
    main()