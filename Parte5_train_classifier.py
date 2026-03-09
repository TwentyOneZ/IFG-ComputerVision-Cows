import os
import glob
import pandas as pd
from pathlib import Path
from pycaret.classification import setup, compare_models, pull

# ==========================================
# CONFIGURAÇÕES
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs" / "pose"
ANALYSIS_DIR = BASE_DIR / "runs" / "analysis"

def ensure_dir(directory):
    directory.mkdir(parents=True, exist_ok=True)

def run_pycaret_automl(df, feature_cols, target_col, scenario_name, out_dir):
    """Roda a suíte completa do PyCaret com Cross-Validation e relatórios interativos no terminal."""
    print(f"\n" + "-"*50)
    print(f"🚀 [INICIANDO: {scenario_name} | {len(feature_cols)} features]")
    print("-"*50)
    
    data = df[feature_cols + [target_col]]
    
    print("⏳ Passo 1/3: Configurando o ambiente do PyCaret (Setup com GPU)...")
    # ---> MUDANÇA CRUCIAL AQUI: use_gpu=True <---
    clf_setup = setup(
        data=data, 
        target=target_col, 
        session_id=42, 
        verbose=True, 
        html=False,
        use_gpu=True  # ⚡ Força algoritmos compatíveis (XGBoost, LightGBM, CatBoost) a usarem a RTX 3070!
    )
    
    print("\n⏳ Passo 2/3: Treinando e comparando modelos (Validação Cruzada)...")
    print("⚠️ Acompanhe o progresso abaixo. A sua RTX 3070 vai acelerar os modelos mais pesados!")
    best_model = compare_models(sort='Accuracy', verbose=True)
    
    print("\n⏳ Passo 3/3: Extraindo resultados finais e salvando as planilhas...")
    results_df = pull()
    
    csv_out = out_dir / f"ranking_{scenario_name}.csv"
    results_df.to_csv(csv_out)
    
    best_model_name = results_df.index[0]
    best_acc = results_df.iloc[0]['Accuracy'] * 100
    print(f"🎉 CONCLUÍDO! Campeão do cenário: {best_model_name} (Acurácia CV: {best_acc:.2f}%)")
    
    return best_model

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
def main():
    print("🤖 Iniciando a Etapa 5: Machine Learning Rigoroso (PyCaret) com Aceleração por GPU")
    ensure_dir(ANALYSIS_DIR)
    
    csv_paths = glob.glob(str(RUNS_DIR / "*" / "extracted_features.csv"))
    
    if not csv_paths:
        print("❌ Nenhuma feature encontrada. Rode a Etapa 3 primeiro!")
        return

    for csv_path in csv_paths:
        model_dir = Path(csv_path).parent
        yolo_model_name = model_dir.name
        print(f"\n\n" + "="*70)
        print(f"⚙️  AVALIANDO A EXTRAÇÃO DO MODELO: {yolo_model_name.upper()}")
        print("="*70)
        
        df = pd.read_csv(csv_path)
        df = df.dropna()
        
        # Garante que o animal_id seja tratado como texto (categoria)
        df['animal_id'] = df['animal_id'].astype(str)
        
        out_dir = ANALYSIS_DIR / yolo_model_name / "machine_learning"
        ensure_dir(out_dir)

        # Filtra as colunas pelas categorias prefixadas
        geom_cols = [c for c in df.columns if c.startswith("geom_")]
        tex_cols = [c for c in df.columns if c.startswith("tex_")]
        all_cols = geom_cols + tex_cols
        
        # Cenário A: Só pelo esqueleto/postura
        run_pycaret_automl(
            df=df, 
            feature_cols=geom_cols, 
            target_col="animal_id", 
            scenario_name="Cenario_A_Apenas_Geometria", 
            out_dir=out_dir
        )
        
        # Cenário B: Esqueleto + Manchas (Pelagem)
        run_pycaret_automl(
            df=df, 
            feature_cols=all_cols, 
            target_col="animal_id", 
            scenario_name="Cenario_B_Geometria_e_Textura", 
            out_dir=out_dir
        )
        
        print(f"\n✅ Rankings do modelo {yolo_model_name} salvos em: {out_dir}")

    print("\n🎊 Etapa 5 concluída com sucesso! Pode abrir a champanhe, os dados estão prontos.")

if __name__ == "__main__":
    main()