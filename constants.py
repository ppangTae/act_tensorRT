from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
DATASET_DIR = parent_dir / "data" / "meloha_box_picking"
CKPT_DIR = parent_dir / "act" / "ckpt"
ONNX_MODEL_DIR = parent_dir / "tensorRT" / "onnx"
ENGINE_DIR = parent_dir / "tensorRT" / "engine"