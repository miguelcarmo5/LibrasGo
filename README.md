# LibrasGo 🤟

Sistema de tradução de Língua Brasileira de Sinais (Libras) em tempo real, utilizando Visão Computacional e Inteligência Artificial. Este projeto é um MVP (Produto Mínimo Viável) que visa quebrar a barreira de comunicação enfrentada pela comunidade surda através de uma solução acessível, escalável e de baixo custo.

## ⚠️ Aviso Importante sobre a Instalação (Utilizadores Windows)

Para evitar bloqueios do **Smart App Control** do Windows 11 e corrupção de ambiente virtual causada pela sincronização do **OneDrive**, é **altamente recomendável** que o projeto seja clonado diretamente para a raiz do disco rígido. 

**NÃO** instale o projeto em pastas como `Downloads`, `Documentos`, `Área de Trabalho` ou dentro do `OneDrive`. 
Recomendamos a criação de uma pasta dedicada, como por exemplo: `C:\Projetos\LibrasGo`.

## 🛠️ Tecnologias Utilizadas
* **Python 3.11** (Obrigatório para compatibilidade estável com o MediaPipe)
* **OpenCV:** Captura de vídeo e processamento de imagem em tempo real.
* **MediaPipe (Google):** Extração de coordenadas geométricas (landmarks) das mãos.
* **Scikit-Learn:** Treino e classificação do modelo de Machine Learning (Random Forest).
* **Pandas:** Estruturação e manipulação da base de dados do modelo.

## ⚙️ Pré-requisitos
* **Python 3.11** instalado e adicionado às variáveis de ambiente (PATH).
* **Git** instalado.
* Uma **Webcam** funcional.

## 🚀 Como Instalar e Configurar

1. **Navegue até à raiz do seu disco (Ex: Disco C) e crie uma pasta segura:**
   ```powershell
   cd C:\
   mkdir Projetos
   cd Projetos
