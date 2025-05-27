import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

# Captura de video (usa 0 si no funciona el 1)
webcam = cv2.VideoCapture(1)

while True:
    exito, imagen = webcam.read()
    if not exito:
        break

    # Voltear y convertir a RGB
    imagen = cv2.flip(imagen, 1)
    img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Procesar imagen
    resultado = hands.process(img_rgb)

    # Dibujar manos si se detectan
    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                imagen,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Proyecto 4 - IA", imagen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
