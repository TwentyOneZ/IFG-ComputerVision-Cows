import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# CONFIGURAÇÕES
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs" / "pose"
ANALYSIS_DIR = BASE_DIR / "runs" / "analysis"

def ensure_dir(directory):
    directory.mkdir(parents=True, exist_ok=True)

def evaluate_and_plot(df, feature_cols, target_col, out_dir, model_name):
    print(f"   Treinando o modelo Campeão (Extra Trees) para {model_name}...")
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Divisão 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Instancia o modelo vencedor
    clf = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Predições no conjunto de teste
    y_pred = clf.predict(X_test)
    
    # ---------------------------------------------------------
    # 1. MATRIZ DE CONFUSÃO
    # ---------------------------------------------------------
    print("   Gerando Matriz de Confusão...")
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title(f"Matriz de Confusão - {model_name} (Extra Trees)", fontsize=16)
    plt.ylabel('Vaca Real')
    plt.xlabel('Vaca Prevista pelo Modelo')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. IMPORTÂNCIA DAS FEATURES (FEATURE IMPORTANCE)
    # ---------------------------------------------------------
    print("   Gerando Gráfico de Importância das Features...")
    importances = clf.feature_importances_
    
    # Cria um dataframe com as importâncias e ordena
    feat_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importancia': importances
    }).sort_values(by='Importancia', ascending=False)
    
    # Pega apenas as 15 mais importantes para o gráfico não ficar gigante
    top_features = feat_df.head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importancia', y='Feature', data=top_features, palette="viridis")
    plt.title(f"Top 15 Features Mais Importantes para Identificar a Vaca - {model_name}", fontsize=14)
    plt.xlabel("Grau de Importância (Gini)")
    plt.ylabel("Feature (Geometria / Textura)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=300)
    plt.close()
    
    # Salva o relatório de classificação completo (Precision, Recall, F1 para CADA vaca)
    report = classification_report(y_test, y_pred, target_names=clf.classes_)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
def main():
    print("📈 Iniciando a Etapa 6: Avaliação Final do Modelo")
    ensure_dir(ANALYSIS_DIR)
    
    # Pega apenas os CSVs do Cenário B (Geometria + Textura), pois sabemos que é o melhor
    csv_paths = glob.glob(str(RUNS_DIR / "*" / "extracted_features.csv"))
    
    if not csv_paths:
        print("❌ Nenhuma feature encontrada. Rode a Etapa 3 primeiro!")
        return

    for csv_path in csv_paths:
        model_dir = Path(csv_path).parent
        yolo_model_name = model_dir.name
        print(f"\n" + "="*60)
        print(f"📊 GERANDO GRÁFICOS PARA: {yolo_model_name}")
        print("="*60)
        
        df = pd.read_csv(csv_path)
        df = df.dropna()
        df['animal_id'] = df['animal_id'].astype(str)
        
        # Vamos rodar a avaliação para o Cenário B (Todas as features)
        geom_cols = [c for c in df.columns if c.startswith("geom_")]
        tex_cols = [c for c in df.columns if c.startswith("tex_")]
        all_cols = geom_cols + tex_cols
        
        out_dir = ANALYSIS_DIR / yolo_model_name / "evaluation"
        ensure_dir(out_dir)
        
        evaluate_and_plot(df, all_cols, "animal_id", out_dir, yolo_model_name)
        
        print(f"✅ Avaliação salva em: {out_dir}")

    print("\n🎉 ETAPA 6 CONCLUÍDA! O pipeline inteiro de pesquisa está finalizado.")

if __name__ == "__main__":
    main()