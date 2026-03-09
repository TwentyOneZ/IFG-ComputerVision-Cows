# 🐄 Top-Down Bovine Biometrics: Pose Estimation & Multimodal Classification

Este repositório contém o código-fonte integral de um *framework* computacional de duplo estágio para a identificação biométrica individual de vacas leiteiras (raça Holandesa) utilizando câmeras de vista superior (*top-down*).

O projeto aborda o desafio da oclusão fenotípica lateral combinando **Estimativa de Pose (YOLOv8)** para mapeamento anatômico espacial e **Machine Learning (PyCaret/Scikit-Learn)** para classificação baseada em fusão multimodal (Geometria + Textura RGB).

---

## ✨ Recursos Principais
* **Pipeline Modular em 7 Etapas:** Do processamento das anotações em formato JSON até a auditoria visual rigorosa de erros de inferência.
* **Avaliação Multi-Arquitetura:** Scripts configurados para testar e comparar modelos de diferentes densidades paramétricas (YOLOv8n, YOLOv8m, YOLO26m).
* **Extração Multimodal de Features:** Cálculo automatizado de proporções corporais, ângulos articulares e amostragem espacial de crominância (RGB) em *patches* localizados.
* **Ablation Study Nativo:** Testes estatísticos rigorosos separando as inferências por geometria pura *versus* geometria espacialmente fundida à textura.
* **Auditoria de Dados com FiftyOne:** Integração nativa com a plataforma FiftyOne para inspeção heurística visual das falsas predições geradas no conjunto de testes.

---

## 📁 Estrutura de Diretórios Esperada

Antes de executar o pipeline, certifique-se de que os seus dados estejam organizados na seguinte estrutura dentro da raiz do projeto:

```text
📦 raiz_do_projeto/
 ┣ 📂 data/
 ┃ ┣ 📂 annotated_data/          # Arquivos JSON gerados pelo Label Studio e imagens originais
 ┃ ┗ 📂 dataset_classificacao/   # Imagens organizadas em subpastas por ID do animal (Ex: /1106/img1.jpg)
 ┣ 📜 .env                       # Variáveis de ambiente (Caminhos, hiperparâmetros de Treino e YOLO)
 ┣ 📜 requirements.txt           # Dependências de ambiente
 ┗ 📜 (Scripts Python Parte 1 a Parte 7)

```

---

## ⚙️ Instalação e Configuração

1. Clone este repositório:

```bash
git clone [https://github.com/SEU_USUARIO/NOME_DO_REPO.git](https://github.com/SEU_USUARIO/NOME_DO_REPO.git)
cd NOME_DO_REPO

```

2. Crie e ative um ambiente virtual isolado (Recomendado):

```bash
python -m venv .venv
# Para Windows:
.venv\Scripts\activate
# Para Linux/Mac:
source .venv/bin/activate

```

3. Instale as dependências computacionais:

```bash
pip install -r requirements.txt

```

*(Obrigatório: Certifique-se de ter a versão correta do PyTorch instalada com suporte a CUDA para viabilizar a aceleração por hardware [GPU] durante o treinamento do YOLO).*

---

## 🚀 Pipeline de Execução (Passo a Passo)

### Estágio 1: Visão Computacional (Estimativa de Pose)

* **`Parte1_build_dataset.py`**
Converte as anotações geradas no *Label Studio* para a topologia nativa do YOLO. Divide o conjunto estratificadamente (Treino 70%, Validação 20%, Teste 10%) e gera o arquivo `dataset_pose.yaml`.
```bash
python Parte1_build_dataset.py

```


* **`Parte2_train_yolo.py`**
Realiza o *fine-tuning* iterativo das redes YOLOv8 (Nano, Medium, etc.) para regressão topológica de 8 pontos-chave da anatomia bovina.
```bash
python Parte2_train_yolo.py

```



### Estágio 2: Extração Multimodal e Machine Learning

* **`Parte3_extract_features.py`**
O motor de extração dimensional. Utiliza os melhores pesos do YOLO para calcular distâncias euclidianas normalizadas, ângulos e janelas de crominância (15x15 pixels), formatando a saída em `extracted_features.csv`.
```bash
python Parte3_extract_features.py

```


* **`Parte4_descriptive_analysis.py`**
Auditoria estatística e exploração de dados univariada e multivariada através da plotagem de Boxplots e Análise de Componentes Principais (PCA).
```bash
python Parte4_descriptive_analysis.py

```


* **`Parte5_train_classifier.py`**
Modelagem preditiva automatizada via **PyCaret**. Avalia algoritmos de decisão (ex: *Extra Trees*, *Random Forest*) e computa o Estudo de Ablação completo entre o Espaço Geométrico e o Espaço Multimodal.
```bash
python Parte5_train_classifier.py

```


* **`Parte6_evaluate_model.py` & `Parte6_evaluate_geometry.py**`
Aferição de desempenho analítico. Plota Matrizes de Confusão de alta resolução e a Importância Paramétrica (*Feature Importance*) para quantificar os tensores responsáveis pela acurácia da rede.
```bash
python Parte6_evaluate_model.py
python Parte6_evaluate_geometry.py

```



### Estágio 3: Diagnóstico Heurístico

* **`Parte7_fiftyone_pycaret.py`**
Avaliação qualitativa orientada à biologia. Emprega a *engine* estrita do Scikit-Learn sobre o isolamento do conjunto de teste e o integra à ferramenta interativa *FiftyOne*, evidenciando visualmente a sobreposição paramétrica que desafia o classificador em instâncias críticas.
```bash
python Parte7_fiftyone_pycaret.py

```



---

## 🔬 Conclusões Científicas do Projeto

1. **O Paradoxo da Complexidade:** Demonstrou-se estatisticamente que arquiteturas de baixa densidade (YOLOv8n) previnem o *overfitting* de coordenadas espaciais de pontos anatômicos melhor do que arquiteturas superparametrizadas (YOLO26m), provendo generalização superior para dados de teste.
2. **Estudo de Ablação (Ruído Geométrico):** Confirmou-se que a variância topológica inter-indivíduos (*top-down*) é estatisticamente insipiente, colapsando classificadores puramente baseados em forma (11% ~ 18%). A ancoragem espacial da crominância RGB reestabeleceu a discriminabilidade, atingindo o limiar de ~73%.
3. **Limitações Fenotípicas Locais:** O *framework* atingiu o teto preditivo imposto pelas homólogas distribuições melanísticas da raça Holandesa. A otimização rumo a acurácias absolutas demanda futuras pesquisas envolvendo vetores espaciais aumentados e redes temporais aplicadas à transição de *frames* em fluxos de vídeo.

---