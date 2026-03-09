import os
import json
import random
import shutil
import urllib.parse
from pathlib import Path

# ==========================================
# CONFIGURAÇÕES
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
SOURCE_DATA_DIR = BASE_DIR / "data" / "annotated_data"
OUTPUT_ROOT = BASE_DIR / "dataset" / "cow_pose_dataset"

SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
RANDOM_SEED = 42

# Ordem exata dos keypoints para o YOLO
KEYPOINT_ORDER = [
    "withers", "back", "hook up", "hook down", 
    "hip", "tail head", "pin down", "pin up"
]

# Mapa de visibilidade para o YOLO (0: não rotulado, 1: oculto, 2: visível)
VISIBILITY_MAP = {
    "visível": 2, "visivel": 2, "visible": 2,
    "oculto": 1, "hidden": 1, 
    "não rotulado": 0, "nao rotulado": 0
}

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================
def extract_target_filename(json_data):
    """Extrai e decodifica o nome do arquivo de imagem do JSON do Label Studio."""
    if isinstance(json_data, list):
        if len(json_data) == 0: return ""
        json_data = json_data[0]
        
    img_path = ""
    # Busca dinamicamente onde está a url da imagem
    if "task" in json_data and "data" in json_data["task"]:
        img_path = json_data["task"]["data"].get("img", "")
    elif "data" in json_data:
        img_path = json_data["data"].get("img", "")
        
    decoded_path = urllib.parse.unquote(img_path)
    filename = decoded_path.replace("\\", "/").split("/")[-1]
    return filename

def get_best_image_match(animal_dir, target_filename):
    """Tenta localizar a imagem real no disco, mesmo que o Label Studio tenha adicionado um hash no nome."""
    # 1. Busca exata
    img_exact = animal_dir / target_filename
    if img_exact.exists() and img_exact.is_file():
        return img_exact
        
    # 2. Busca ignorando o prefixo hash (ex: '1a2b3c4d-arquivo.jpg' -> 'arquivo.jpg')
    if "-" in target_filename:
        possible_name = target_filename.split("-", 1)[-1]
        img_no_hash = animal_dir / possible_name
        if img_no_hash.exists() and img_no_hash.is_file():
            return img_no_hash
            
    # 3. Busca elástica (o nome do disco está contido no alvo, ou vice-versa)
    for img in animal_dir.iterdir():
        if img.is_file() and img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            if img.name in target_filename or target_filename in img.name:
                return img
                
    return None

def convert_labelstudio_to_yolo(json_data):
    """Converte os dados do JSON para uma linha no formato YOLO Pose e calcula BBox se não existir."""
    if isinstance(json_data, list):
        if len(json_data) == 0: return None
        json_data = json_data[0]

    results = json_data.get("result", [])
    if not results:
        return None
    
    bbox = None
    keypoints_dict = {}
    visibility_dict = {}

    for item in results:
        item_type = item.get("type")
        val = item.get("value", {})
        item_id = item.get("id")

        if item_type == "rectanglelabels" and "cow" in val.get("rectanglelabels", []):
            x_tl = val["x"] / 100.0
            y_tl = val["y"] / 100.0
            w = val["width"] / 100.0
            h = val["height"] / 100.0
            x_center = x_tl + (w / 2.0)
            y_center = y_tl + (h / 2.0)
            bbox = [x_center, y_center, w, h]

        elif item_type == "keypointlabels":
            kp_name = val.get("keypointlabels", [""])[0]
            x = val["x"] / 100.0
            y = val["y"] / 100.0
            keypoints_dict[item_id] = {"name": kp_name, "x": x, "y": y}
            
        elif item_type == "choices":
            choice = val.get("choices", [""])[0].lower()
            visibility_dict[item_id] = VISIBILITY_MAP.get(choice, 2)

    # =========================================================
    # FALLBACK: Se o anotador marcou apenas pontos e nenhuma bounding box!
    if not bbox:
        if not keypoints_dict:
            return None # Não há vaca, não há pontos. Ignora.
        
        # Cria uma "caixa" baseada nos limites de onde os pontos estão + 2% de folga
        xs = [k["x"] for k in keypoints_dict.values()]
        ys = [k["y"] for k in keypoints_dict.values()]
        min_x, max_x = max(0.0, min(xs) - 0.02), min(1.0, max(xs) + 0.02)
        min_y, max_y = max(0.0, min(ys) - 0.02), min(1.0, max(ys) + 0.02)
        
        w = max_x - min_x
        h = max_y - min_y
        x_center = min_x + w / 2.0
        y_center = min_y + h / 2.0
        bbox = [x_center, y_center, w, h]

    # Monta a linha: <class_id> <x> <y> <w> <h> <px1> <py1> <pv1> ...
    yolo_line = f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"

    for kp_target in KEYPOINT_ORDER:
        kp_found = False
        for k_id, k_data in keypoints_dict.items():
            if k_data["name"] == kp_target:
                x, y = k_data["x"], k_data["y"]
                v = visibility_dict.get(k_id, 2)
                yolo_line += f" {x:.6f} {y:.6f} {v}"
                kp_found = True
                break
        
        if not kp_found:
            yolo_line += " 0.000000 0.000000 0"

    return yolo_line

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
def main():
    print("Iniciando conversão do dataset...")
    
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    for split in SPLIT_RATIOS.keys():
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    valid_pairs = []
    missing_images = []

    for animal_dir in SOURCE_DATA_DIR.iterdir():
        if not animal_dir.is_dir(): continue
            
        kp_dirs = [d for d in animal_dir.iterdir() if d.is_dir() and d.name.lower() == "key_points"]
        if not kp_dirs: continue
            
        kp_dir = kp_dirs[0]
        
        for json_file in kp_dir.iterdir():
            if not json_file.is_file(): continue
            
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                target_filename = extract_target_filename(data)
                if not target_filename:
                    continue
                    
                # Aqui entra a inteligência nova de localizar arquivos
                img_path = get_best_image_match(animal_dir, target_filename)
                
                if img_path:
                    yolo_content = convert_labelstudio_to_yolo(data)
                    if yolo_content:
                        valid_pairs.append({
                            "img_src": img_path,
                            "yolo_txt": yolo_content,
                            "out_stem": img_path.stem
                        })
                else:
                    missing_images.append(f"JSON {json_file.name} referenciou a imagem '{target_filename}', que não foi encontrada na pasta '{animal_dir.name}'.")
            except Exception as e:
                print(f"Erro ao processar anotação {json_file}: {e}")

    random.seed(RANDOM_SEED)
    random.shuffle(valid_pairs)
    
    total = len(valid_pairs)
    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])

    splits = {
        "train": valid_pairs[:train_end],
        "val": valid_pairs[train_end:val_end],
        "test": valid_pairs[val_end:]
    }

    for split, pairs in splits.items():
        for pair in pairs:
            img_dst = OUTPUT_ROOT / "images" / split / pair["img_src"].name
            shutil.copy2(pair["img_src"], img_dst)
            
            txt_dst = OUTPUT_ROOT / "labels" / split / f"{pair['out_stem']}.txt"
            with open(txt_dst, "w", encoding="utf-8") as f:
                f.write(pair["yolo_txt"] + "\n")

    yaml_content = f"""path: {OUTPUT_ROOT.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: cow

kpt_shape: [{len(KEYPOINT_ORDER)}, 3]
flip_idx: [] 
"""
    with open(OUTPUT_ROOT / "dataset_pose.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("-" * 40)
    print(f"✅ Conversão concluída com sucesso!")
    print(f"Imagens processadas: {total}")
    print(f"  - Treino (Train): {len(splits['train'])}")
    print(f"  - Validação (Val): {len(splits['val'])}")
    print(f"  - Teste (Test): {len(splits['test'])}")
    print(f"Dataset salvo em: {OUTPUT_ROOT}")
    
    if missing_images:
         print(f"⚠️ Atenção: Não conseguimos achar a imagem física para {len(missing_images)} anotações JSON.")

if __name__ == "__main__":
    main()