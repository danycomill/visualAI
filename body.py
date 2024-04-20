import cv2
import mediapipe as mp

# Inicializar los módulos de MediaPipe para detección de manos, cuerpo y objetos
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_objectron = mp.solutions.objectron

hands = mp_hands.Hands()
pose = mp_pose.Pose()
objectron = mp_objectron.Objectron()

# Configurar la captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir la imagen de BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar el frame con MediaPipe para detección de manos
    results_hands = hands.process(frame_rgb)
    
    # Procesar el frame con MediaPipe para detección de cuerpo
    results_pose = pose.process(frame_rgb)
    
    # Procesar el frame con MediaPipe para detección de objetos (celulares)
    results_objectron = objectron.process(frame_rgb)
    
    # Dibujar el cuerpo detectado
    if results_pose.pose_landmarks:
        # Dibujar líneas conectando los puntos del cuerpo
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Marcar cada esquina con puntos
        for lm in results_pose.pose_landmarks.landmark:
            ih, iw, _ = frame.shape
            x, y = int(lm.x * iw), int(lm.y * ih)
            cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)
    
    # Mostrar resultados de detección de manos
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Dibujar líneas conectando los puntos de los dedos
            connections = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20]]
            for connection in connections:
                x0, y0 = int(hand_landmarks.landmark[connection[0]].x * frame.shape[1]), int(hand_landmarks.landmark[connection[0]].y * frame.shape[0])
                x1, y1 = int(hand_landmarks.landmark[connection[1]].x * frame.shape[1]), int(hand_landmarks.landmark[connection[1]].y * frame.shape[0])
                cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            
            # Contar el número de dedos levantados
            fingers_up = 0
            for idx, lm in enumerate(hand_landmarks.landmark):
                # Verificar si los dedos están levantados
                if idx in [4, 8, 12, 16, 20] and lm.y < hand_landmarks.landmark[idx - 3].y:
                    fingers_up += 1
            
            # Mostrar el número de dedos levantados
            cv2.putText(frame, f"Dedos: {fingers_up}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Dibujar cuadros delimitadores alrededor de los celulares detectados
    if results_objectron.detected_objects:
        for detected_object in results_objectron.detected_objects:
            bbox = detected_object.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Mostrar el frame
    cv2.imshow('Detector de Manos, Cuerpo y Celulares', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
