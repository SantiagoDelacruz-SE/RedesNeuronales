import cv2
import numpy as np
import os


def load_and_preprocess_image(image_path, target_size=(64, 64)):
    """
    Carga una imagen, la redimensiona y normaliza sus píxeles.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB
        img = cv2.resize(img, target_size)
        img = img.astype("float32") / 255.0  # Normalizar a [0, 1]
        return img
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None


def extract_color_features(image_array):
    """
    Extrae características de color simples (promedio de R, G, B) de una imagen.
    Esta es una simplificación; en un proyecto real, se usarían técnicas más robustas.
    """
    if image_array is None:
        return None
    # Calcular el promedio de los canales R, G, B
    avg_color = np.mean(image_array, axis=(0, 1))
    return avg_color


def create_dummy_dataset(num_samples_per_class=20):
    """
    Crea un dataset simulado de características y etiquetas.
    En un escenario real, esto cargaría tus imágenes de semáforo.
    """
    print("Creando dataset de prueba simulado...")
    features = []
    labels = []
    class_mapping = {"red": 0, "yellow": 1, "green": 2}

    # Simular características de color para cada clase
    # Rojo: Más rojo
    for _ in range(num_samples_per_class):
        f = np.random.uniform(low=[0.7, 0.1, 0.1], high=[1.0, 0.3, 0.3])
        features.append(f)
        labels.append(class_mapping["red"])

    # Amarillo: Más amarillo (rojo + verde)
    for _ in range(num_samples_per_class):
        f = np.random.uniform(low=[0.7, 0.7, 0.1], high=[1.0, 1.0, 0.3])
        features.append(f)
        labels.append(class_mapping["yellow"])

    # Verde: Más verde
    for _ in range(num_samples_per_class):
        f = np.random.uniform(low=[0.1, 0.7, 0.1], high=[0.3, 1.0, 0.3])
        features.append(f)
        labels.append(class_mapping["green"])

    features = np.array(features)
    labels = np.array(labels)

    print(f"Dataset simulado creado: {len(features)} muestras.")
    return features, labels, class_mapping


def load_real_dataset(data_dir, class_labels=["red", "yellow", "green"], target_size=(64, 64)):
    """
    Carga imágenes desde directorios específicos para cada clase y extrae características.
    data_dir debería contener subdirectorios como 'red/', 'yellow/', 'green/'.
    """
    all_features = []
    all_labels = []
    class_mapping = {label: i for i, label in enumerate(class_labels)}

    print(f"Cargando dataset desde: {data_dir}")
    for label_name in class_labels:
        class_path = os.path.join(data_dir, label_name)
        if not os.path.isdir(class_path):
            print(f"Advertencia: Directorio {class_path} no encontrado. Saltando {label_name}.")
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img_array = load_and_preprocess_image(img_path, target_size)
            if img_array is not None:
                features = extract_color_features(img_array)
                if features is not None:
                    all_features.append(features)
                    all_labels.append(class_mapping[label_name])

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    print(f"Dataset cargado: {len(all_features)} muestras.")
    return all_features, all_labels, class_mapping


def process_raw_images_to_processed(raw_data_dir="data/raw", processed_data_dir="data/processed", target_size=(64, 64)):
    """
    Lee imágenes de la carpeta raw, las preprocesa y las clasifica en
    las subcarpetas 'red', 'yellow', 'green' dentro de 'data/processed/'.

    La clasificación se basa en una heurística simple de los colores RGB promedio
    de la imagen preprocesada.

    Args:
        raw_data_dir (str): Directorio de las imágenes originales.
        processed_data_dir (str): Directorio donde se guardarán las imágenes procesadas.
        target_size (tuple): Tamaño al que se redimensionarán las imágenes.
    """
    print(f"Iniciando procesamiento de imágenes de {raw_data_dir} a {processed_data_dir}...")

    # Asegúrate de que los directorios de salida existan
    output_dirs = {
        "red": os.path.join(processed_data_dir, "red"),
        "yellow": os.path.join(processed_data_dir, "yellow"),
        "green": os.path.join(processed_data_dir, "green")
    }
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)

    processed_count = {"red": 0, "yellow": 0, "green": 0, "unclassified": 0}

    # Recorrer todas las imágenes en el directorio raw
    for img_name in os.listdir(raw_data_dir):
        img_path = os.path.join(raw_data_dir, img_name)

        # Ignorar directorios o archivos no-imagen
        if not os.path.isfile(img_path) or not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            continue

        print(f"Procesando: {img_name}")
        # Cargar y preprocesar la imagen
        processed_img_array = load_and_preprocess_image(img_path, target_size)

        if processed_img_array is not None:
            # Extraer características de color (promedio RGB)
            features = extract_color_features(processed_img_array)

            if features is not None:
                r, g, b = features[0], features[1], features[2]  # Desempaquetar los promedios RGB

                # --- Lógica de CLASIFICACIÓN BASADA EN UMBRALES DE COLOR ---
                # ADVERTENCIA: Esta es una heurística simple. Los valores umbral
                # pueden necesitar ajustes finos dependiendo de la iluminación,
                # la saturación de los colores de tus semáforos, etc.
                # También considera convertir a HSV para una mejor discriminación de color.

                classified_label = "unclassified"

                # Puedes ajustar estos umbrales para que se adapten a tus imágenes
                # Por ejemplo, para rojo: el canal rojo es alto, los otros son bajos.
                # Asegúrate de que la luz del semáforo sea el color dominante.

                print(f"  RGB promedio para {img_name}: R={r:.2f}, G={g:.2f}, B={b:.2f}")

                # Semáforo Rojo
                if r > g * 1.2 and r > b * 1.2 and r > 0.4:  # Rojo significativamente más alto que verde y azul
                    classified_label = "red"
                # Semáforo Verde
                elif g > r * 1 and g > b * 1 and g > 0.4:  # Verde significativamente más alto que rojo y azul
                    classified_label = "green"
                # Semáforo Amarillo (una mezcla de rojo y verde)
                elif r > 0.4 and g > 0.4 and b < 0.3 and (
                        abs(r - g) < 0.3):  # Rojo y verde altos, azul bajo, y son cercanos
                    classified_label = "yellow"

                # Si una imagen de semáforo no cae en estas categorías, podría ser un problema
                # con los umbrales o la calidad de la imagen.

                if classified_label != "unclassified":
                    # Convertir el array normalizado de vuelta a 0-255 y a BGR para guardar con OpenCV
                    img_to_save = (processed_img_array * 255).astype(np.uint8)
                    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)

                    # Generar un nuevo nombre de archivo único para evitar sobrescribir
                    # Usamos un contador para asegurar nombres únicos dentro de cada clase
                    current_idx = processed_count[classified_label]
                    new_img_name = f"{classified_label}_{current_idx:04d}{os.path.splitext(img_name)[1]}"
                    output_path = os.path.join(output_dirs[classified_label], new_img_name)

                    cv2.imwrite(output_path, img_to_save)
                    processed_count[classified_label] += 1
                    print(f"  Clasificado como '{classified_label}' y guardado en: {output_path}")
                else:
                    processed_count["unclassified"] += 1
                    print(
                        f"  No se pudo clasificar '{img_name}'. Puede que necesites ajustar los umbrales de color o revisar la imagen.")
            else:
                processed_count["unclassified"] += 1
                print(f"  No se pudieron extraer características de color para '{img_name}'.")
        else:
            processed_count["unclassified"] += 1
            print(f"  No se pudo preprocesar '{img_name}'.")

    print("\n--- Resumen del Procesamiento ---")
    print(f"Total imágenes en '{raw_data_dir}': {len(os.listdir(raw_data_dir))}")
    for label, count in processed_count.items():
        print(f"  Clasificadas '{label}': {count}")
    print("Procesamiento de imágenes RAW a PROCESSED completado.")


def start_camera_feed(prediction_callback=None, camera_index=0):
    """
    Inicia la captura de video de la cámara y procesa los frames.

    Args:
        prediction_callback (function, optional): Una función que toma un frame procesado
                                                 y retorna una predicción. Por defecto None.
        camera_index (int): El índice de la cámara a usar (0 para la predeterminada,
                            puede ser 1, 2, etc., o la URL de DroidCam).
    """
    # Si usas DroidCam, la URL típica es algo como 'http://192.168.1.XX:4747/video'
    # Descomenta y ajusta si es tu caso:
    # cam = cv2.VideoCapture('http://192.168.1.XX:4747/video')
    cam = cv2.VideoCapture(camera_index)

    if not cam.isOpened():
        print(f"Error: No se pudo abrir la cámara con índice/URL {camera_index}")
        return

    print(f"Cámara con índice/URL {camera_index} abierta.")

    # Dimensiones de la ventana para mostrar la ROI procesada
    roi_display_size = (200, 200)  # Tamaño de la ventana de la ROI

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: No se pudo obtener el frame. ¿Cámara desconectada o error de lectura?")
            break

        # Mostrar el frame original de la cámara (se actualizará con el texto de la predicción)

        # --- Procesamiento del Frame para el Modelo ---
        # Definir una ROI (Región de Interés) fija como ejemplo.
        # AJUSTA ESTOS VALORES para que la caja verde contenga tu semáforo real.
        # Valores de ejemplo (ajustar según tu cámara y donde aparezca el semáforo):
        # roi_y_start, roi_y_end = int(frame.shape[0] * 0.2), int(frame.shape[0] * 0.8)
        # roi_x_start, roi_x_end = int(frame.shape[1] * 0.3), int(frame.shape[1] * 0.7)

        # Ejemplo de ROI para un semáforo que esté en la parte superior central de la imagen
        roi_y_start = int(frame.shape[0] * 0.1)  # 10% desde arriba
        roi_y_end = int(frame.shape[0] * 0.4)  # 40% desde arriba
        roi_x_start = int(frame.shape[1] * 0.4)  # 40% desde la izquierda
        roi_x_end = int(frame.shape[1] * 0.6)  # 60% desde la izquierda

        # DIBUJAR RECTANGULO DE LA ROI EN LA VENTANA PRINCIPAL
        cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)

        # Asegúrate de que la ROI es válida
        if roi_y_start < roi_y_end and roi_x_start < roi_x_end:
            traffic_light_roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            if traffic_light_roi.size > 0:  # Asegúrate de que la ROI no esté vacía
                # Convertir ROI a formato adecuado para el modelo (RGB, normalizado, redimensionado)
                processed_roi_rgb = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2RGB)
                # Redimensionar la ROI para el modelo y para visualización consistente
                processed_roi_rgb_resized = cv2.resize(processed_roi_rgb, (64, 64))
                processed_roi_normalized = processed_roi_rgb_resized.astype("float32") / 255.0

                # Extraer características
                current_features = extract_color_features(processed_roi_normalized)

                # Mostrar la ROI procesada en una ventana separada
                # Redimensionar para la ventana de visualización si es necesario
                display_roi = cv2.resize(processed_roi_rgb, roi_display_size)
                cv2.imshow("Semáforo ROI Procesada", display_roi)

                # Si se proporciona una función de predicción, úsala
                prediction_text = "Procesando..."
                if prediction_callback and current_features is not None:
                    # current_features es un array 1D, el modelo espera 2D (ej. [[R, G, B]])
                    model_input = np.expand_dims(current_features, axis=0)
                    prediction_text = prediction_callback(model_input)

                # Muestra la predicción en la ventana principal de la cámara
                cv2.putText(frame, f"Estado: {prediction_text}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                prediction_text = "ROI Semáforo Vacía"
                cv2.putText(frame, prediction_text, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            prediction_text = "ROI Semáforo Inválida"
            cv2.putText(frame, prediction_text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Actualiza la ventana de la cámara original con la predicción (siempre al final)
        cv2.imshow("Camara Original", frame)

        # Esperar por la tecla 'ESC' (27) para salir
        key = cv2.waitKey(1) & 0xFF  # waitKey(1) para video
        if key == 27:
            print("Saliendo de la cámara.")
            break

    cam.release()
    cv2.destroyAllWindows()