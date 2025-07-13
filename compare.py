import os
import sys
import torch
from torch.nn import functional as F
import time
from itertools import cycle
from polygraphy.backend import trt as poly_trt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/home/park/ros2/act/act")
from act_pytorch import make_act_pytorch_model
from act.act.utils import load_data

##### 변수 설정 #####
dataset_dir = "/home/park/ros2/tensorRT/meloha_box_picking_data"
ckpt_path = "/home/park/ros2/act/ckpt/policy_best.ckpt"
camera_names = ['cam_head', 'cam_left_wrist','cam_right_wrist']
batch_size_train = 9
batch_size_val = 9
chunk_size = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### 데이터 준비 #####
train_dataloader, val_dataloader, stats, _ = load_data(
    dataset_dir,
    camera_names,
    batch_size_train,
    batch_size_val,
)

# ── 원하는 설정 ─────────────────────────────────────────
num_sample = 40        # 최종적으로 쌓을 샘플 수
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
    print("="*50)
    print(f"act pytorch model 평균 L1 loss : {act_pytorch_mean_l1_loss}")
    print(f"act pytorch model 평균 추론 시간: {act_pytorch_mean_infer_time * 1000:.3f} ms")
    print("="*50)

# # * static Quantization 적용된 tensorRT engine accuarcy 및 inference time 측정

# Load TensorRT engine from ENGINE_PATH.
ENGINE_PATH = "/home/park/ros2/tensorRT/act_origin.engine"
engine = poly_trt.engine_from_bytes(open(ENGINE_PATH, "rb").read())

total_infer_time = []
all_l1_loss = []

with poly_trt.TrtRunner(engine) as runner:
    for idx in range(num_sample):

        image = images_all[idx:idx+1]
        qpos = qpos_all[idx:idx+1]
        actions = actions_all[idx:idx+1]
        is_pad = is_pad_all[idx:idx+1]

        before_infer = time.time()
        output_dict = runner.infer({"qpos" : qpos, "image" : image})
        after_infer = time.time()
        inference_time = after_infer - before_infer
        total_infer_time.append(inference_time)
        actions_hat = output_dict["action"] # shape : (1, 60, 6)

        actions = actions[:, :chunk_size, :]
        is_pad = is_pad[:, :chunk_size]
        all_l1 = F.l1_loss(actions, actions_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        all_l1_loss.append(l1)

# 평균 L1 loss와 평균 추론 시간 계산
act_origin_mean_l1_loss = torch.stack(all_l1_loss).mean().item()
act_origin_mean_infer_time = torch.tensor(total_infer_time).mean().item()
print("="*50)
print(f"act origin model 평균 L1 loss : {act_origin_mean_l1_loss}")
print(f"act origin model 평균 추론 시간: {act_origin_mean_infer_time * 1000:.3f} ms")
print("="*50)

# # * static Quantization 적용안된 tensorRT engine accuarcy 및 inference time 측정

# Load TensorRT engine from ENGINE_PATH.
ENGINE_PATH = "/home/park/ros2/tensorRT/act_int8_fp16_tf32.engine"
engine = poly_trt.engine_from_bytes(open(ENGINE_PATH, "rb").read())

total_infer_time = []
all_l1_loss = []

with poly_trt.TrtRunner(engine) as runner:
    for idx in range(num_sample):

        image = images_all[idx:idx+1]
        qpos = qpos_all[idx:idx+1]
        actions = actions_all[idx:idx+1]
        is_pad = is_pad_all[idx:idx+1]

        before_infer = time.time()
        output_dict = runner.infer({"qpos" : qpos, "image" : image})
        after_infer = time.time()
        inference_time = after_infer - before_infer
        total_infer_time.append(inference_time)
        actions_hat = output_dict["action"] # shape : (1, 60, 6)

        actions = actions[:, :chunk_size, :]
        is_pad = is_pad[:, :chunk_size]
        all_l1 = F.l1_loss(actions, actions_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        all_l1_loss.append(l1)

# 평균 L1 loss와 평균 추론 시간 계산
act_int8_fp16_tf32_mean_l1_loss = torch.stack(all_l1_loss).mean().item()
act_int8_fp16_tf32_mean_infer_time = torch.tensor(total_infer_time).mean().item()
print("="*50)
print(f"act_int8_fp16_tf32 평균 L1 loss : {act_int8_fp16_tf32_mean_l1_loss}")
print(f"act_int8_fp16_tf32 평균 추론 시간: {act_int8_fp16_tf32_mean_infer_time * 1000:.3f} ms")
print("="*50)

# # * static Quantization 적용안된 tensorRT engine accuarcy 및 inference time 측정

# Load TensorRT engine from ENGINE_PATH.
ENGINE_PATH = "/home/park/ros2/tensorRT/act_no_calib_int8_fp16_tf32.engine"
engine = poly_trt.engine_from_bytes(open(ENGINE_PATH, "rb").read())

total_infer_time = []
all_l1_loss = []

with poly_trt.TrtRunner(engine) as runner:
    for idx in range(num_sample):

        image = images_all[idx:idx+1]
        qpos = qpos_all[idx:idx+1]
        actions = actions_all[idx:idx+1]
        is_pad = is_pad_all[idx:idx+1]

        before_infer = time.time()
        output_dict = runner.infer({"qpos" : qpos, "image" : image})
        after_infer = time.time()
        inference_time = after_infer - before_infer
        total_infer_time.append(inference_time)
        actions_hat = output_dict["action"] # shape : (1, 60, 6)

        actions = actions[:, :chunk_size, :]
        is_pad = is_pad[:, :chunk_size]
        all_l1 = F.l1_loss(actions, actions_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        all_l1_loss.append(l1)

# 평균 L1 loss와 평균 추론 시간 계산
act_no_calib_int8_fp16_tf32_mean_l1_loss = torch.stack(all_l1_loss).mean().item()
act_no_calib_int8_fp16_tf32_mean_infer_time = torch.tensor(total_infer_time).mean().item()
print("="*50)
print(f"act_no_calib_int8_fp16_tf32 평균 L1 loss : {act_no_calib_int8_fp16_tf32_mean_l1_loss}")
print(f"act_no_calib_int8_fp16_tf32 평균 추론 시간: {act_no_calib_int8_fp16_tf32_mean_infer_time * 1000:.3f} ms")
print("="*50)