import math
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


ARQUIVO_DADOS = Path(__file__).with_name("dados_maos_normalizados.csv")
ARQUIVO_MODELO = Path(__file__).with_name("modelo_libras.pkl")
COLUNA_LABEL = "label"
NUMERO_FEATURES = 63


def main():
	if not ARQUIVO_DADOS.exists():
		raise FileNotFoundError(f"Arquivo de dados nao encontrado: {ARQUIVO_DADOS}")

	dados = pd.read_csv(ARQUIVO_DADOS, encoding="utf-8-sig")
	if COLUNA_LABEL not in dados.columns:
		raise ValueError(f"O CSV precisa conter a coluna '{COLUNA_LABEL}'.")
	if dados.empty:
		raise ValueError("O CSV nao contem amostras para treinamento.")

	X = dados.drop(columns=[COLUNA_LABEL])
	y = dados[COLUNA_LABEL]

	if X.shape[1] != NUMERO_FEATURES:
		raise ValueError(
			f"Esperadas {NUMERO_FEATURES} coordenadas, mas o CSV tem {X.shape[1]}."
		)
	if y.isna().any() or X.isna().any().any():
		raise ValueError("O CSV contem valores vazios; corrija os dados antes do treino.")

	classes = y.nunique()
	if classes < 2:
		raise ValueError(
			"O CSV precisa ter amostras de pelo menos duas letras. "
			f"No momento, so existe: {', '.join(map(str, y.unique()))}."
		)

	tamanho_teste = math.ceil(len(dados) * 0.2)
	tamanho_treino = len(dados) - tamanho_teste
	if tamanho_teste < classes or tamanho_treino < classes or (y.value_counts() < 2).any():
		raise ValueError(
			"Sao necessarias pelo menos duas amostras por letra para dividir "
			"os dados entre treino e teste."
		)

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	modelo = RandomForestClassifier(n_estimators=200, random_state=42)
	modelo.fit(X_train, y_train)

	previsoes = modelo.predict(X_test)
	precisao = accuracy_score(y_test, previsoes)
	with ARQUIVO_MODELO.open("wb") as arquivo:
		pickle.dump(modelo, arquivo)

	print(f"Acuracia no teste: {precisao:.2%}")
	print(f"Modelo salvo em: {ARQUIVO_MODELO}")


if __name__ == "__main__":
	main()
