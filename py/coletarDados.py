import argparse
import csv
import math
from pathlib import Path

import cv2
import mediapipe as mp


def criar_cabecalho():
	colunas = ["label"]
	for indice in range(21):
		colunas.extend([f"x{indice}", f"y{indice}", f"z{indice}"])
	return colunas


def extrair_pontos_normalizados(hand_landmarks):
	pulso = hand_landmarks[0]
	ponto_palma = hand_landmarks[9]
	tamanho_palma = math.sqrt(
		(ponto_palma.x - pulso.x) ** 2
		+ (ponto_palma.y - pulso.y) ** 2
		+ (ponto_palma.z - pulso.z) ** 2
	)

	if tamanho_palma < 1e-6:
		return None

	pontos = []
	for landmark in hand_landmarks:
		pontos.extend([
			(landmark.x - pulso.x) / tamanho_palma,
			(landmark.y - pulso.y) / tamanho_palma,
			(landmark.z - pulso.z) / tamanho_palma,
		])
	return pontos


def main():
	parser = argparse.ArgumentParser(description="Coleta landmarks de uma mao para treinamento.")
	parser.add_argument("label", help="Nome do sinal coletado, por exemplo A ou ola")
	args = parser.parse_args()

	arquivo_modelo = Path(__file__).with_name("hand_landmarker.task")
	arquivo_dados = Path(__file__).with_name("dados_maos_normalizados.csv")

	arquivo_novo = not arquivo_dados.exists() or arquivo_dados.stat().st_size == 0
	with arquivo_dados.open("a", newline="", encoding="utf-8") as arquivo:
		escritor = csv.writer(arquivo)
		if arquivo_novo:
			escritor.writerow(criar_cabecalho())

		opcoes = mp.tasks.vision.HandLandmarkerOptions(
			base_options=mp.tasks.BaseOptions(model_asset_path=str(arquivo_modelo)),
			running_mode=mp.tasks.vision.RunningMode.VIDEO,
			num_hands=1,
			min_hand_detection_confidence=0.5,
			min_hand_presence_confidence=0.5,
			min_tracking_confidence=0.5,
		)

		camera = cv2.VideoCapture(0)
		if not camera.isOpened():
			raise RuntimeError("Nao foi possivel abrir a camera.")

		frame_id = 0
		total_amostras = 0
		with mp.tasks.vision.HandLandmarker.create_from_options(opcoes) as landmarker:
			while True:
				sucesso, frame = camera.read()
				if not sucesso:
					continue

				frame = cv2.flip(frame, 1)
				imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)
				resultado = landmarker.detect_for_video(imagem_mp, frame_id * 33)
				frame_id += 1

				mensagem = f"Sinal: {args.label} | Amostras: {total_amostras} | ESPACO salva | Q sai"
				cv2.putText(
					frame,
					mensagem,
					(20, 35),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.65,
					(0, 255, 0),
					2,
				)

				if resultado.hand_landmarks:
					pontos = extrair_pontos_normalizados(resultado.hand_landmarks[0])
					altura, largura, _ = frame.shape
					for landmark in resultado.hand_landmarks[0]:
						centro = (int(landmark.x * largura), int(landmark.y * altura))
						cv2.circle(frame, centro, 5, (0, 0, 255), -1)

				cv2.imshow("Coleta de dados Libras", frame)
				tecla = cv2.waitKey(1) & 0xFF
				if tecla == ord(" ") and pontos is not None:
					escritor.writerow([args.label, *pontos])
					arquivo.flush()
					total_amostras += 1
				if tecla == ord("q"):
					break

		camera.release()
		cv2.destroyAllWindows()
		print(f"{total_amostras} amostras salvas em {arquivo_dados}")


if __name__ == "__main__":
	main()
