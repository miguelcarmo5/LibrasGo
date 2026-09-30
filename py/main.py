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

def draw_hand_landmarks(frame, hand_landmarks):
    height, width = frame.shape[:2]

    for start, end in HAND_CONNECTIONS:
        point_start = hand_landmarks[start]
        point_end = hand_landmarks[end]
        position_start = (int(point_start.x * width), int(point_start.y * height))
        position_end = (int(point_end.x * width), int(point_end.y * height))
        cv2.line(frame, position_start, position_end, (0, 255, 0), 2)

    for landmark in hand_landmarks:
        center = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, center, 5, (0, 0, 255), -1)


def main():
    cap = cv2.VideoCapture(0)

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            print("Iniciando câmara com deteção de mãos... Prima 'q' para sair.")
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int(time.time() * 1000)
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                for hand_landmarks in detection_result.hand_landmarks:
                    draw_hand_landmarks(frame, hand_landmarks)

                cv2.imshow("Deteccao Libras - MediaPipe Tasks", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()