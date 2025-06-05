from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.image_processor import (
    create_dummy_dataset,
    load_real_dataset,
    start_camera_feed,
    extract_color_features,
    load_and_preprocess_image,
    process_raw_images_to_processed
)
from models.mlp_classifier import MLPTrafficLightClassifier
from models.som_model import SOMTrafficLightClassifier
from models.rbf_classifier import RBFNetwork
from models.hopfield_network import HopfieldNetwork

CLASS_LABELS = {0: "red", 1: "yellow", 2: "green"}
SIGNAL_MAP = {
    "red": "ADVERTENCIA (ALTO)",
    "yellow": "PELIGRO (PRECAUCION)",
    "green": "SEGURIDAD (PUEDE CRUZAR)",
    "desconocido": "Estado desconocido"
}


def get_signal_message(predicted_label):
    if predicted_label in SIGNAL_MAP:
        return SIGNAL_MAP[predicted_label]
    elif isinstance(predicted_label, (int, np.integer)) and predicted_label in CLASS_LABELS:
        text_label = CLASS_LABELS[predicted_label]
        return SIGNAL_MAP.get(text_label, "ESTADO DESCONOCIDO")
    else:
        return SIGNAL_MAP["desconocido"]


def run_mlp_prediction(model, class_names, scaler=None):
    class_index_to_name = {i: name for i, name in enumerate(class_names)}

    def predict_frame(frame_features):
        if scaler:
            frame_features = scaler.transform(frame_features)
        predicted_class_index = model.predict(frame_features)[0]
        predicted_label_name = class_index_to_name[predicted_class_index]
        return get_signal_message(predicted_label_name)
    return predict_frame


def run_som_prediction(model, class_names, scaler=None):
    def predict_frame(frame_features):
        if scaler:
            frame_features_scaled = scaler.transform(frame_features)[0]
        else:
            frame_features_scaled = frame_features[0]
        predicted_label = model.predict(frame_features_scaled)
        return get_signal_message(predicted_label)
    return predict_frame


def run_rbf_prediction(model, class_names, scaler=None):
    class_index_to_name = {i: name for i, name in enumerate(class_names)}
    def predict_frame(frame_features):
        if scaler:
            frame_features = scaler.transform(frame_features) # Asegúrate de que el scaler esté ajustado (fitted)
        predicted_class_index = model.predict(frame_features)[0]
        predicted_label_name = class_index_to_name[predicted_class_index]
        return get_signal_message(predicted_label_name)
    return predict_frame


def run_hopfield_prediction(model, stored_patterns_dict):
    def binarize_for_hopfield(features):
        r, g, b = features[0]
        binary_pattern = np.full((3, 3), -1, dtype=int).flatten()
        if r > g and r > b and r > 0.6:
            binary_pattern[4] = 1
        elif g > r and g > b and g > 0.6:
            binary_pattern[7] = 1
        elif r > 0.5 and g > 0.5 and b < 0.3 and (abs(r - g) < 0.2):
            binary_pattern[1] = 1
        return binary_pattern

    def classify_hopfield_output(recalled_pattern, stored_patterns_dict):
        min_diff = float('inf')
        best_match_label = "desconocido"
        for stored_p_tuple, label in stored_patterns_dict.items():
            stored_p_array = np.array(stored_p_tuple)
            diff = model.compare_patterns(recalled_pattern, stored_p_array)
            if diff < min_diff:
                min_diff = diff
                best_match_label = label
        return best_match_label, min_diff

    def predict_frame(frame_features):
        binary_input_pattern = binarize_for_hopfield(frame_features)
        recalled_pattern = model.recall(binary_input_pattern, max_iter=50)
        predicted_label, _ = classify_hopfield_output(recalled_pattern, stored_patterns_dict)
        return get_signal_message(predicted_label)
    return predict_frame


def main():
    print("Iniciando el sistema de observación de semáforos...")

    raw_data_directory = "data/raw"
    processed_data_directory = "data/processed"
    target_img_size = (64, 64)

    print("\n##### PREPROCESANDO IMÁGENES RAW A PROCESSED #####")
    process_raw_images_to_processed(raw_data_directory, processed_data_directory, target_img_size) #

    print("\n##### CARGANDO DATASET PARA ENTRENAMIENTO DESDE 'PROCESSED' #####")
    class_labels_list = ["red", "yellow", "green"]
    X, y, class_mapping = load_real_dataset(processed_data_directory, class_labels_list, target_img_size) #

    if len(X) == 0:
        print("Error: No se encontraron datos de semáforos procesados para cargar.")
        return

    class_names = list(class_mapping.keys()) #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) #

    print(f"\nDatos preparados. Clases: {class_names}")
    print(f"Tamaño del dataset de entrenamiento: {len(X_train)} muestras") #
    print(f"Tamaño del dataset de prueba: {len(X_test)} muestras") #

    # Variables para modelos y scalers (se inicializarán según la selección)
    mlp_classifier = None
    scaler_mlp = None
    som_model = None
    scaler_som = None
    rbf_net = None
    scaler_rbf = None
    hopfield_net = None
    hopfield_pattern_labels = None

    while True:
        print("\n--- MENÚ DE SELECCIÓN DE RED NEURONAL ---")
        print("1. Entrenar y usar MLP")
        print("2. Entrenar y usar SOM")
        print("3. Entrenar y usar RBF")
        print("4. Entrenar y usar Hopfield")
        print("5. Salir")

        choice = input("Selecciona una opción (1-5): ")

        camera_prediction_func = None #

        if choice == '1':
            print("\n##### ENTRENANDO MLP #####") #
            scaler_mlp = StandardScaler() #
            X_train_scaled_mlp = scaler_mlp.fit_transform(X_train) #
            X_test_scaled_mlp = scaler_mlp.transform(X_test) #
            mlp_classifier = MLPTrafficLightClassifier(hidden_layer_sizes=(50, 25), max_iter=1000) #
            mlp_classifier.train(X_train_scaled_mlp, y_train) #
            mlp_classifier.evaluate(X_test_scaled_mlp, y_test, class_names) #
            print("\n¡Iniciando cámara con predicciones de MLP! Presiona ESC para salir.") #
            camera_prediction_func = run_mlp_prediction(mlp_classifier.model, class_names, scaler=scaler_mlp) #

        elif choice == '2':
            print("\n##### ENTRENANDO SOM #####") #
            scaler_som = MinMaxScaler() #
            X_scaled_som_train = scaler_som.fit_transform(X_train) #
            X_scaled_som_all = scaler_som.transform(X) # # Para el mapeo de neuronas
            som_model = SOMTrafficLightClassifier(x=10, y=10, input_len=X_scaled_som_train.shape[1], sigma=0.5, learning_rate=0.5) #
            som_model.train(X_scaled_som_train, num_iteration=20000) #
            som_model.classify_map(X_scaled_som_all, y, class_names) #
            # som_model.visualize_map(X_scaled_som_all, y, class_names) # Descomentar para visualización
            print("\n¡Iniciando cámara con predicciones de SOM! Presiona ESC para salir.") #
            camera_prediction_func = run_som_prediction(som_model, class_names, scaler=scaler_som) #


        elif choice == '3':
            print("\n##### ENTRENANDO RED RBF #####") #
            scaler_rbf = StandardScaler() #
            X_scaled_rbf_train = scaler_rbf.fit_transform(X_train) #
            X_scaled_rbf_test = scaler_rbf.transform(X_test) #
            rbf_net = RBFNetwork(n_rbf_neurons=10, gamma=0.5) #
            rbf_net.train(X_scaled_rbf_train, y_train) #
            rbf_net.evaluate(X_scaled_rbf_test, y_test, class_names) #
            print("\n¡Iniciando cámara con predicciones de RBF! Presiona ESC para salir.") #
            # ¡IMPORTANTE! Asegúrate que scaler_rbf esté ajustado (fitted) antes de pasarlo aquí.
            # Ya se hizo con scaler_rbf.fit_transform(X_train)
            camera_prediction_func = run_rbf_prediction(rbf_net, class_names, scaler=scaler_rbf) #

        elif choice == '4':
            print("\n##### ENTRENANDO RED HOPFIELD #####") #
            pattern_red = np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]]).flatten() #
            pattern_yellow = np.array([[-1, 1, -1], [-1, -1, -1], [-1, -1, -1]]).flatten() #
            pattern_green = np.array([[-1, -1, -1], [-1, -1, -1], [-1, 1, -1]]).flatten() #
            stored_hopfield_patterns = [pattern_red, pattern_yellow, pattern_green] #
            hopfield_pattern_labels = { #
                tuple(pattern_red): "red", #
                tuple(pattern_yellow): "yellow", #
                tuple(pattern_green): "green" #
            }
            hopfield_net = HopfieldNetwork(num_neurons=len(pattern_red)) #
            hopfield_net.train(stored_hopfield_patterns) #
            print("\n¡Iniciando cámara con predicciones de Hopfield! Presiona ESC para salir.") #
            camera_prediction_func = run_hopfield_prediction(hopfield_net, hopfield_pattern_labels) #

        elif choice == '5':
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, intenta de nuevo.")
            continue

        if camera_prediction_func:
            start_camera_feed(prediction_callback=camera_prediction_func, camera_index=0) #
            # Al salir de la cámara, cv2.destroyAllWindows() se llama dentro de start_camera_feed
            # Podemos volver al menú principal o decidir terminar.
            # Por ahora, volverá al menú. Si quieres salir después de la cámara, añade un break aquí.

    print("\nPrograma finalizado.")
    cv2.destroyAllWindows() # # Asegurar que todas las ventanas se cierren al final


if __name__ == "__main__":
    main()