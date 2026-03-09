import os
import pandas as pd
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict
import fiftyone as fo

# ==========================================
# CONFIGURAÇÕES
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
# Caminho para as pastas organizadas das vacas (usado na Parte 3)
CLASSIFICATION_DIR = BASE_DIR / "data" / "dataset_classificacao" 
# Caminho direto para o CSV do modelo campeão (YOLO26m)
CSV_PATH = BASE_DIR / "runs" / "pose" / "yolov8n-pose_ep100_sz960_b8_rectT" / "extracted_features.csv"

def main():
    print("👀 Iniciando a Etapa 7: Análise Visual de Erros com FiftyOne")
    
    if not CSV_PATH.exists():
        print(f"❌ Erro: Não encontrei o CSV do YOLO26m em {CSV_PATH}")
        return

    print("📊 Carregando features extraídas e gerando predições para todo o dataset...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()
    df['animal_id'] = df['animal_id'].astype(str)

    # Usa TODAS as features (Cenário B - Geometria + Textura)
    geom_cols = [c for c in df.columns if c.startswith("geom_")]
    tex_cols = [c for c in df.columns if c.startswith("tex_")]
    feature_cols = geom_cols + tex_cols

    X = df[feature_cols]
    y = df["animal_id"]

    # Instancia o modelo campeão
    clf = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # Cria um pipeline que primeiro normaliza os dados (como o PyCaret faz) e depois classifica
    clf = make_pipeline(StandardScaler(), ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1))

    y_pred = cross_val_predict(clf, X, y, cv=5)    
    # Faz predições para 100% das imagens usando validação cruzada (K-Fold=5)
    # Isso garante que a predição foi feita de forma "justa" (sem a imagem estar no treino)
    y_pred = cross_val_predict(clf, X, y, cv=5)
    
    df['predicted_id'] = y_pred
    df['is_correct'] = df['animal_id'] == df['predicted_id']

    print("🖼️ Construindo o dataset no FiftyOne...")
    
    # Limpa o dataset se você já tiver rodado este script antes
    dataset_name = "Cow_Classification_Analysis"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
        
    dataset = fo.Dataset(dataset_name)
    samples = []
    missing_images = 0

    # Adiciona cada imagem ao painel do FiftyOne
    for idx, row in df.iterrows():
        # Constrói o caminho real da imagem usando o diretório dataset_classificacao
        img_path = CLASSIFICATION_DIR / str(row['animal_id']) / str(row['image_name'])
        
        if not img_path.exists():
            missing_images += 1
            continue

        sample = fo.Sample(filepath=str(img_path))
        
        # Etiqueta de Verdade Absoluta (Gabarito)
        sample["ground_truth"] = fo.Classification(label=str(row['animal_id']))
        # Etiqueta do que o nosso Modelo de IA previu
        sample["prediction"] = fo.Classification(label=str(row['predicted_id']))
        # Status de acerto para facilitar o filtro visual
        sample["is_correct"] = row['is_correct']
        
        samples.append(sample)

    dataset.add_samples(samples)
    
    if missing_images > 0:
        print(f"⚠️ Atenção: {missing_images} imagens não foram encontradas no caminho esperado.")

    print(f"\n🎉 Tudo pronto! Abrindo o FiftyOne com {len(samples)} imagens.")
    print("👉 DICA: No painel à esquerda do FiftyOne, expanda 'is_correct' e selecione 'False' para ver só os erros!")
    
    # Inicia a interface web
    session = fo.launch_app(dataset)
    session.wait() # Mantém o script rodando para o servidor não desligar

if __name__ == "__main__":
    main()