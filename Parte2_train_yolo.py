import os
import json
import torch
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

def main():
    print("Carregando configurações do .env...")
    load_dotenv()
    
    # ==========================================
    # VERIFICAÇÃO DE HARDWARE (GPU vs CPU)
    # ==========================================
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detectada: {gpu_name} (Memória VRAM: {vram_gb:.2f} GB)")
        device = "0" # Indica para o YOLO usar a primeira GPU (índice 0)
    else:
        print("⚠️ Nenhuma GPU com suporte a CUDA detectada pelo PyTorch!")
        print("O treinamento ocorrerá na CPU e será muito mais lento.")
        print("Dica: Certifique-se de ter instalado o PyTorch com suporte a CUDA (https://pytorch.org/get-started/locally/).")
        device = "cpu"
    # ==========================================

    # Busca o caminho do YAML do dataset
    dataset_yaml = os.getenv("DATASET_YAML", "dataset/cow_pose_dataset/dataset_pose.yaml")
    yaml_path = Path(dataset_yaml).resolve()
    
    if not yaml_path.exists():
        print(f"❌ Erro: Arquivo YAML não encontrado em {yaml_path}")
        return

    # Lê e converte a configuração dos modelos do .env
    models_config_str = os.getenv("MODELS_CONFIG", "[]")
    try:
        models_config = json.loads(models_config_str)
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao decodificar MODELS_CONFIG do .env. Verifique a sintaxe JSON. Erro: {e}")
        return

    if not models_config:
        print("⚠️ Nenhum modelo configurado no .env (MODELS_CONFIG está vazio).")
        return

    # Diretório raiz onde todas as comparações serão salvas
    base_project_dir = Path("runs/pose").resolve()

    print(f"\n🚀 Iniciando bateria de testes com {len(models_config)} configurações de modelos...")
    print("=" * 60)

    # Itera sobre cada configuração definida no .env
    for config in models_config:
        model_name = config.get("model", "yolov8n-pose.pt")
        epochs = config.get("epochs", 100)
        imgsz = config.get("imgsz", 640)
        batch = config.get("batch", 16)
        rect = config.get("rect", True)
        
        # ---> NOVOS PARÂMETROS DE AUGMENTAÇÃO <---
        degrees = config.get("degrees", 0.0)
        fliplr = config.get("fliplr", 0.5)
        flipud = config.get("flipud", 0.0)
        
        # Gera um nome de pasta único e padronizado
        safe_model_name = model_name.replace(".pt", "").replace("/", "_").replace("\\", "_")
        rect_str = "T" if rect else "F"
        run_name = f"{safe_model_name}_ep{epochs}_sz{imgsz}_b{batch}_rect{rect_str}"
        
        print(f"\n▶️ Iniciando experimento: {run_name}")
        print(f"   Parâmetros: {config}")
        print(f"   Processamento: GPU (device={device})" if device == "0" else "   Processamento: CPU")
        
        try:
            model = YOLO(model_name)
            
            # Inicia o treinamento com controle total da geometria da imagem
            results = model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                rect=rect,
                degrees=degrees, # Rotação
                fliplr=fliplr,   # Espelhamento horizontal
                flipud=flipud,   # Espelhamento vertical
                device=device,  
                project=str(base_project_dir), 
                name=run_name,                 
                exist_ok=True,
                plots=True
            )
            
            print(f"✅ Treinamento de {run_name} concluído!")
            
            # Avaliação final automática
            print(f"📊 Avaliando métricas finais para {run_name}...")
            model.val(device=device) # Faz a validação na GPU também
            
            print(f"💾 Resultados e gráficos salvos em: {base_project_dir / run_name}")
            
        except Exception as e:
            print(f"\n❌ Erro crítico ao treinar o modelo {model_name}:")
            print(e)
            
            # Captura amigável para estouro de memória de vídeo
            if "CUDA out of memory" in str(e):
                print("\n💡 DICA DE HARDWARE: Sua RTX 3070 ficou sem VRAM (Out of Memory).")
                print(f"Isso ocorreu porque a combinação de imagem grande (imgsz={imgsz}) com o tamanho do lote (batch={batch}) excedeu os 8GB.")
                print(f"Solução: Vá no arquivo .env e reduza o 'batch' do {model_name} pela metade (ex: de {batch} para {max(1, int(batch/2))}).")
            
            print("\nPausando este experimento e indo para o próximo (se houver)...")
            # Libera a memória da GPU para não prejudicar o próximo modelo do loop
            if torch.cuda.is_available():
                torch.cuda.empty_cache() 
            
    print("\n" + "=" * 60)
    print("🎉 Bateria de treinamentos finalizada!")

if __name__ == '__main__':
    # Proteção para multiprocessamento no Windows
    main()