import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Diccionario de gestos y su significado
signos = {
    'puño_con_pulgar_arriba': 'Si',
    'puño_con_pulgar_abajo': 'No',
    'dos_manos': 'Gracias',
    'una_mano_abierta': 'Hola',
    'rock_con_pulgar_arriba': 'Te amo'
}

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB y detectar las manos
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Dibujar los puntos de las manos si están presentes
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obtener la posición de la muñeca y el dedo pulgar
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

            # Calcular la distancia entre la muñeca y el dedo pulgar
            distance_thumb_index = ((wrist.x - thumb.x)**2 + (wrist.y - thumb.y)**2)**0.5
            distance_thumb_pinky = ((wrist.x - pinky.x)**2 + (wrist.y - pinky.y)**2)**0.5

            # Determinar el gesto
            if distance_thumb_index < 0.1:
                signo_actual = 'puño_con_pulgar_arriba'
            elif distance_thumb_pinky < 0.1:
                signo_actual = 'puño_con_pulgar_abajo'
            elif index.y < middle.y and index.y < ring.y:
                signo_actual = 'rock_con_pulgar_arriba'
            elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                signo_actual = 'dos_manos'
            elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
                signo_actual = 'una_mano_abierta'
            else:
                continue

            # Mostrar el significado del gesto en pantalla
            cv2.putText(frame, signos[signo_actual], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostrar la imagen
    cv2.imshow('Lenguaje de señas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
