import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURAÇÕES DA ANÁLISE
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs" / "pose"
ANALYSIS_DIR = BASE_DIR / "runs" / "analysis"

# Filtro: Número de vacas para plotar no gráfico. 
# (Definido como 10 para não poluir visualmente, mas pode aumentar se quiser)
TOP_N_COWS = 10 

def ensure_dir(directory):
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# FUNÇÕES DE PLOTAGEM
# ==========================================
def plot_boxplots(df, top_cows, out_dir, model_name):
    """Gera boxplots para mostrar a variância intra e inter-classes das features geométricas."""
    df_top = df[df["animal_id"].isin(top_cows)].copy()
    
    features_to_plot = ["geom_angle_withers_back_hip", "geom_dist_hook_width"]
    titles = ["Ângulo: Cernelha-Dorso-Quadril", "Proporção: Largura do Quadril"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Distribuição de Features Geométricas por Animal - Modelo: {model_name}", fontsize=16)

    for i, feature in enumerate(features_to_plot):
        if feature in df_top.columns:
            # Usando 'husl' para gerar cores distintas para cada vaca
            sns.boxplot(data=df_top, x="animal_id", y=feature, ax=axes[i], palette="husl")
            axes[i].set_title(titles[i])
            axes[i].set_xlabel("ID do Animal")
            axes[i].set_ylabel("Valor")
            axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(out_dir / f"boxplot_geometry.png", dpi=300)
    plt.close()

def plot_correlation(df, out_dir, model_name):
    """Gera um mapa de calor das correlações das features geométricas."""
    geom_cols = [c for c in df.columns if c.startswith("geom_")]
    corr = df[geom_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Matriz de Correlação das Features Geométricas - {model_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"correlation_matrix.png", dpi=300)
    plt.close()

def plot_pca(df, top_cows, out_dir, model_name):
    """Gera gráficos de dispersão PCA (Cenário A: Só Geometria | Cenário B: Geo + Textura)."""
    df_top = df[df["animal_id"].isin(top_cows)].copy()
    
    geom_cols = [c for c in df.columns if c.startswith("geom_")]
    tex_cols = [c for c in df.columns if c.startswith("tex_")]
    all_cols = geom_cols + tex_cols

    scenarios = {
        "PCA_A_Only_Geometry": geom_cols,
        "PCA_B_Geometry_and_Texture": all_cols
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Projeção PCA 2D (Agrupamento por Animal) - {model_name}", fontsize=16)

    for i, (scenario_name, cols) in enumerate(scenarios.items()):
        X = df_top[cols].values
        y = df_top["animal_id"].values

        # Padroniza os dados (z-score) antes do PCA
        X_scaled = StandardScaler().fit_transform(X)
        
        # Reduz para 2 dimensões
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Gráfico - usando paleta 'husl' para suportar mais de 10 cores perfeitamente
        sns.scatterplot(
            x=X_pca[:, 0], y=X_pca[:, 1], 
            hue=y, palette="husl", ax=axes[i], 
            s=80, alpha=0.8
        )
        var_expl = pca.explained_variance_ratio_ * 100
        axes[i].set_title(f"{scenario_name.replace('_', ' ')}\nVar. Explicada: {var_expl[0]:.1f}% e {var_expl[1]:.1f}%")
        axes[i].set_xlabel(f"Componente Principal 1")
        axes[i].set_ylabel(f"Componente Principal 2")
        axes[i].legend(title="Animal ID", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(out_dir / f"pca_clusters.png", dpi=300)
    plt.close()

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
def main():
    print("📊 Iniciando a Etapa 4: Análise Descritiva das Features")
    
    ensure_dir(ANALYSIS_DIR)
    
    csv_paths = glob.glob(str(RUNS_DIR / "*" / "extracted_features.csv"))
    
    if not csv_paths:
        print("❌ Nenhum extracted_features.csv encontrado. Rode a Etapa 3 primeiro!")
        return

    for csv_path in csv_paths:
        model_dir = Path(csv_path).parent
        model_name = model_dir.name
        print(f"\n⚙️ Processando análise para o modelo: {model_name}")
        
        df = pd.read_csv(csv_path)
        
        # ---> AJUSTE CRUCIAL: Forçar o ID da vaca a ser tratado como Texto/Categoria <---
        df['animal_id'] = df['animal_id'].astype(str)
        
        out_dir = ANALYSIS_DIR / model_name
        ensure_dir(out_dir)

        top_cows = df["animal_id"].value_counts().nlargest(TOP_N_COWS).index.tolist()
        
        print(f"   Gerando Boxplots...")
        plot_boxplots(df, top_cows, out_dir, model_name)
        
        print(f"   Gerando Matriz de Correlação...")
        plot_correlation(df, out_dir, model_name)
        
        print(f"   Calculando e projetando PCA...")
        plot_pca(df, top_cows, out_dir, model_name)
        
        print(f"✅ Gráficos salvos em: {out_dir}")

    print("\n🎉 Etapa 4 concluída! Abra os arquivos .png na pasta 'runs/analysis/' para visualizar os resultados.")

if __name__ == "__main__":
    main()