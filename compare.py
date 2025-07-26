import os
import sys
import torch
from torch.nn import functional as F
import time
from itertools import cycle
from polygraphy.backend import trt as poly_trt
from pathlib import Path
from tqdm import tqdm

parent_dir = Path(__file__).resolve().parent.parent
act_dir = parent_dir / "act" / "act"
sys.path.append(str(parent_dir))
sys.path.append(str(act_dir))

from act_pytorch import make_act_pytorch_model
from act.act.utils import load_data

##### 변수 설정 #####
parent_dir = Path(__file__).resolve().parent.parent
dataset_dir = parent_dir / "data" / "meloha_box_picking"
ckpt_dir = parent_dir / "act" / "ckpt"
ckpt_path = parent_dir / "act" / "ckpt" / "policy_best.ckpt"
camera_names = ['cam_head', 'cam_left_wrist','cam_right_wrist']
batch_size_train = 9
batch_size_val = 9
chunk_size = 30 # !
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### 데이터 준비 #####
train_dataloader, val_dataloader, stats, _ = load_data(
    str(dataset_dir),
    camera_names,
    batch_size_train,
    batch_size_val,
)

# ── 원하는 설정 ─────────────────────────────────────────
num_sample = 4000        # 최종적으로 쌓을 샘플 수
batch_size = 9          # dataloader 한 번에 나오는 샘플 수
num_batches  = (num_sample + batch_size - 1) // batch_size  # 5
# ───────────────────────────────────────────────────────

# stack 할 임시 버퍼
imgs, qposes, acts, pads = [], [], [], []   # 리스트 4개 준비

# dataloader를 무한 반복할 수 있게 cycle로 감싼다.
val_iter = cycle(val_dataloader)
for idx in range(num_batches):
    image, qpos, actions, is_pad = next(val_iter)
    imgs.append(image)
    qposes.append(qpos)
    acts.append(actions)
    pads.append(is_pad)

# cat 이후 초과 부분(40개 초과)을 잘라낸다.
images_all  = torch.cat(imgs,  dim=0)[:num_sample]
qpos_all   = torch.cat(qposes, dim=0)[:num_sample]
actions_all = torch.cat(acts,  dim=0)[:num_sample]
is_pad_all  = torch.cat(pads,  dim=0)[:num_sample]

def main():
    # act_pytorch accuaracy 및 inference time 측정
    act_pytorch = make_act_pytorch_model()
    loading_status = act_pytorch.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    act_pytorch.cuda()
    act_pytorch.eval()

    total_infer_time = []
    all_l1_loss = []

    with torch.no_grad():
        for idx in range(num_sample):
            image = images_all[idx:idx+1]
            qpos = qpos_all[idx:idx+1]
            actions = actions_all[idx:idx+1]
            is_pad = is_pad_all[idx:idx+1]

            before_infer = time.time()
            actions_hat = act_pytorch(qpos.cuda(), image.cuda()).cpu()
            after_infer = time.time()
            inference_time = after_infer - before_infer
            total_infer_time.append(inference_time)

            actions = actions[:, :chunk_size, :]
            is_pad = is_pad[:, :chunk_size]
            all_l1 = F.l1_loss(actions, actions_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            all_l1_loss.append(l1)

        # 평균 L1 loss와 평균 추론 시간 계산
        act_pytorch_mean_l1_loss = torch.stack(all_l1_loss).mean().item()
        act_pytorch_mean_infer_time = torch.tensor(total_infer_time).mean().item()
        print("\n" + "=" * 60)
        print(f"act pytorch model | {act_pytorch_mean_l1_loss:10.5f} | {act_pytorch_mean_infer_time*1000:14.3f}ms")
        print("=" * 60)

    # ── 1. 비교할 엔진 이름(또는 전체 경로)만 나열 ──────────────────────
    engine_names = [
        "act_qat.engine",
        # "act_qat_enc4_dec7_chunk60.engine",
        "act_ptq_int8.engine",
        "act_qat_fp16.engine"
        # "act_ptq_int8_enc4_dec7_chunk60.engine",
    ]

    # ── 2. 엔진이 들어 있는 기본 디렉터리 설정 ────────────────────────────
    engine_dir = parent_dir / "tensorRT" / "engine"

    results = []  # [(engine_name, mean_l1, mean_ms), ...]

    for name in tqdm(engine_names, desc="Evaluating engines"):
        path = Path(name) if Path(name).is_absolute() else engine_dir / name
        if not path.exists():
            print(f"[Warning] 엔진 파일이 없음: {path}")
            continue

        mean_l1, mean_ms = evaluate_engine(path)
        results.append((name, mean_l1, mean_ms))

    # ── 5. 결과 출력 ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Engine':35s} | {'Mean L1':>10s} | {'Mean Time (ms)':>14s}")
    print("-" * 60)
    for name, l1, ms in results:
        print(f"{name:35s} | {l1:10.5f} | {ms:14.3f}")
    print("=" * 60)

def evaluate_engine(engine_path: Path):
    """ 엔진 하나를 로드해서 평균 L1 loss / 추론시간(ms)을 반환 """
    engine_bytes = engine_path.read_bytes()
    engine = poly_trt.engine_from_bytes(engine_bytes)

    total_infer_time, all_l1_loss = [], []

    with poly_trt.TrtRunner(engine) as runner:
        for idx in range(num_sample):
            image  = images_all[idx : idx + 1]
            qpos   = qpos_all[idx : idx + 1]
            gt_act = actions_all[idx : idx + 1]
            is_pad = is_pad_all[idx : idx + 1]

            # ── 추론 ────────────────────────────────────────────────
            t0 = time.time()
            pred = runner.infer({"qpos": qpos, "image": image})
            t1 = time.time()

            total_infer_time.append(t1 - t0)

            # ── L1 loss (패딩 제외) ────────────────────────────────
            pred_act = pred["action"]                # (1, 30, 6)
            gt_act   = gt_act[:, :chunk_size, :]
            is_pad   = is_pad[:, :chunk_size]

            l1_all = F.l1_loss(gt_act, pred_act, reduction="none")
            l1     = (l1_all * ~is_pad.unsqueeze(-1)).mean()
            all_l1_loss.append(l1)

    mean_l1   = torch.stack(all_l1_loss).mean().item()
    mean_time = torch.tensor(total_infer_time).mean().item() * 1000  # ms
    return mean_l1, mean_time



if __name__ == "__main__":
    main()