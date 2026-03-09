import os
import pandas as pd
from pathlib import Path
import fiftyone as fo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# CONFIGURAÇÕES
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

# A sua organização correta de pastas
CLASSIFICATION_DIR = BASE_DIR / "data" / "dataset_classificacao" 

# Apontando para o campeão: YOLOv8n
CSV_PATH = BASE_DIR / "runs" / "pose" / "yolov8n-pose_ep100_sz960_b8_rectT" / "extracted_features.csv"

def main():
    print("👀 Iniciando a Etapa 7: Análise Visual do Conjunto de Teste (Motor Scikit-Learn Puro)")
    
    if not CSV_PATH.exists():
        print(f"❌ Erro: Não encontrei o CSV em {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    df = df.dropna()
    df['animal_id'] = df['animal_id'].astype(str)

    geom_cols = [c for c in df.columns if c.startswith("geom_")]
    tex_cols = [c for c in df.columns if c.startswith("tex_")]
    feature_cols = geom_cols + tex_cols
    
    X = df[feature_cols]
    y = df['animal_id']

    # ==========================================
    # RECRIANDO A DIVISÃO EXATA DO PYCARET
    # ==========================================
    print("⚙️ Passo 1/3: Recriando a divisão (70% Treino / 30% Teste)...")
    # test_size=0.3 e random_state=42 garantem as mesmas imagens de teste que o PyCaret usou
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("🌳 Passo 2/3: Treinando o Extra Trees Classifier...")
    # Parâmetros padrão recriando o comportamento vitorioso
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("🔮 Passo 3/3: Gerando as predições de teste...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Acurácia alcançada neste conjunto de teste: {acc*100:.2f}%")

    # Criamos um DataFrame apenas com os resultados do teste
    df_test = pd.DataFrame({
        'image_name': df.loc[X_test.index, 'image_name'],
        'animal_id': y_test,
        'predicted_id': y_pred
    })
    df_test['is_correct'] = df_test['animal_id'] == df_test['predicted_id']

    # ==========================================
    # MONTANDO O FIFTYONE
    # ==========================================
    print("🖼️ Construindo o dataset no FiftyOne (Apenas Imagens de Teste)...")
    
    dataset_name = "Cow_Classification_TestSet"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
            
    dataset = fo.Dataset(dataset_name)
    samples = []
    missing_images = 0

    for idx, row in df_test.iterrows():
        img_path = CLASSIFICATION_DIR / str(row['animal_id']) / str(row['image_name'])
        
        if not img_path.exists():
            missing_images += 1
            continue

        sample = fo.Sample(filepath=str(img_path))
        
        sample["ground_truth"] = fo.Classification(label=str(row['animal_id']))
        sample["prediction"] = fo.Classification(label=str(row['predicted_id']))
        sample["is_correct"] = row['is_correct']
        
        samples.append(sample)

    dataset.add_samples(samples)
    
    if missing_images > 0:
        print(f"⚠️ Aviso: {missing_images} imagens não foram encontradas na pasta.")

    print(f"\n🎉 Tudo pronto! Foram carregadas {len(samples)} imagens de teste.")
    print("\n👉 DICA PARA SUA ANÁLISE:")
    print("No lado esquerdo do FiftyOne, clique em 'is_correct' e selecione 'False'.")
    
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    main()