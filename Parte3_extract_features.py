import os
import cv2
import math
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# CONFIGURAÇÕES
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

# A pasta onde estão os IDs das vacas e suas 50 imagens
CLASSIFICATION_DIR = BASE_DIR / "data" / "dataset_classificacao" 
RUNS_DIR = BASE_DIR / "runs" / "pose"

KP_NAMES = [
    "withers", "back", "hook_up", "hook_down", 
    "hip", "tail_head", "pin_down", "pin_up"
]

PATCH_SIZE = 15

# ==========================================
# FUNÇÕES MATEMÁTICAS
# ==========================================
def calc_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calc_angle(p1, p2, p3):
    ang = math.degrees(math.atan2(p3[1]-p2[1], p3[0]-p2[0]) - math.atan2(p1[1]-p2[1], p1[0]-p2[0]))
    return ang + 360 if ang < 0 else ang

def get_color_features(image, center_x, center_y, patch_size):
    h, w, _ = image.shape
    x, y = int(center_x), int(center_y)
    
    x1, x2 = max(0, x - patch_size//2), min(w, x + patch_size//2)
    y1, y2 = max(0, y - patch_size//2), min(h, y + patch_size//2)
    
    patch = image[y1:y2, x1:x2]
    if patch.size == 0: return 0, 0, 0, 0, 0, 0
        
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    mean_color = np.mean(patch, axis=(0, 1))
    std_color = np.std(patch, axis=(0, 1))
    
    return mean_color[0], mean_color[1], mean_color[2], std_color[0], std_color[1], std_color[2]

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
def main():
    print("🔎 Iniciando a Etapa 3: Inferência em massa no Dataset de Classificação")
    
    if not CLASSIFICATION_DIR.exists():
        print(f"❌ Pasta {CLASSIFICATION_DIR} não encontrada!")
        return

    # Acha os modelos treinados (YOLOv8n, YOLOv8m, etc)
    model_paths = glob.glob(str(RUNS_DIR / "*" / "weights" / "best.pt"))
    if not model_paths:
        print("❌ Nenhum modelo treinado encontrado na pasta runs!")
        return

    for model_path in model_paths:
        model_dir = Path(model_path).parent.parent
        model_name = model_dir.name
        print(f"\n🧠 Carregando modelo treinado: {model_name}")
        
        try:
            model = YOLO(model_path)
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            continue

        dataset_features = []

        # Entra em cada pasta de animal (ex: 1106, 1122...)
        for animal_folder in CLASSIFICATION_DIR.iterdir():
            if not animal_folder.is_dir(): continue
            
            animal_id = animal_folder.name
            
            # Pega todas as imagens JPG daquela vaca
            for img_path in animal_folder.glob("*.jpg"):
                img_cv = cv2.imread(str(img_path))
                if img_cv is None: continue

                # O Robô 1 entra em ação: Adivinha onde estão os pontos!
                results = model.predict(source=img_cv, verbose=False, conf=0.5)
                
                # Se o modelo não achou vaca na imagem com >50% de confiança, pula
                if len(results) == 0 or len(results[0].keypoints) == 0:
                    continue 
                    
                boxes = results[0].boxes
                keypoints = results[0].keypoints.data[0] 
                
                kp_dict = {}
                for i, name in enumerate(KP_NAMES):
                    x, y, conf = keypoints[i].tolist()
                    kp_dict[name] = (x, y, conf)

                # Começa a montar a linha da planilha
                feat_row = {
                    "animal_id": animal_id, 
                    "image_name": img_path.name,
                    "model_used": model_name
                }

                # --- 1. GEOMETRIA ---
                body_length = calc_distance(kp_dict["withers"], kp_dict["tail_head"])
                body_length = body_length if body_length > 0 else 1.0 

                feat_row["geom_dist_hook_width"] = calc_distance(kp_dict["hook_up"], kp_dict["hook_down"]) / body_length
                feat_row["geom_dist_pin_width"] = calc_distance(kp_dict["pin_up"], kp_dict["pin_down"]) / body_length
                feat_row["geom_dist_withers_back"] = calc_distance(kp_dict["withers"], kp_dict["back"]) / body_length
                feat_row["geom_dist_back_hip"] = calc_distance(kp_dict["back"], kp_dict["hip"]) / body_length

                feat_row["geom_angle_withers_back_hip"] = calc_angle(kp_dict["withers"], kp_dict["back"], kp_dict["hip"])
                feat_row["geom_angle_back_hip_tail"] = calc_angle(kp_dict["back"], kp_dict["hip"], kp_dict["tail_head"])
                feat_row["geom_angle_hookU_hip_hookD"] = calc_angle(kp_dict["hook_up"], kp_dict["hip"], kp_dict["hook_down"])

                # --- 2. TEXTURA E COR ---
                for kp_name, (x, y, _) in kp_dict.items():
                    r, g, b, sr, sg, sb = get_color_features(img_cv, x, y, PATCH_SIZE)
                    feat_row[f"tex_{kp_name}_R_mean"] = r
                    feat_row[f"tex_{kp_name}_G_mean"] = g
                    feat_row[f"tex_{kp_name}_B_mean"] = b

                x1, y1, x2, y2 = boxes.xyxy[0].tolist() 
                box_patch = img_cv[int(y1):int(y2), int(x1):int(x2)]
                if box_patch.size > 0:
                    box_patch = cv2.cvtColor(box_patch, cv2.COLOR_BGR2RGB)
                    mean_bbox = np.mean(box_patch, axis=(0, 1))
                    feat_row["tex_bbox_R_mean"] = mean_bbox[0]
                    feat_row["tex_bbox_G_mean"] = mean_bbox[1]
                    feat_row["tex_bbox_B_mean"] = mean_bbox[2]
                else:
                    for color in ["R", "G", "B"]: feat_row[f"tex_bbox_{color}_mean"] = 0

                dataset_features.append(feat_row)

        df = pd.DataFrame(dataset_features)
        csv_path = model_dir / f"extracted_features.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Tabela gerada em: {csv_path} (Vacas processadas: {len(df)})")

    print("\n🎉 Etapa 3 concluída! Os dados estão prontos para classificação.")

if __name__ == "__main__":
    main()