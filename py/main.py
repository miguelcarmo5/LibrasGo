import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Conexões clássicas entre os 21 pontos da mão para traçar as linhas
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    print("Iniciando câmara com deteção de mãos... Prima 'q' para sair.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Timestamp em milissegundos exigido pelo modo VIDEO
        timestamp_ms = int(time.time() * 1000)
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w, _ = frame.shape

        # Desenho manual dos pontos e ligações
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Desenhar as linhas entre as articulações
                for connection in HAND_CONNECTIONS:
                    pt1 = hand_landmarks[connection[0]]
                    pt2 = hand_landmarks[connection[1]]
                    p1 = (int(pt1.x * w), int(pt1.y * h))
                    p2 = (int(pt2.x * w), int(pt2.y * h))
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)

                # Desenhar os círculos em cada um dos 21 pontos
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        cv2.imshow("Deteccao Libras - MediaPipe Tasks", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()