import os
import sys
import time

from PIL import Image
from polygraphy.backend import trt as poly_trt
import tqdm
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
from pprint import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/home/park/ros2/act/act")
from act.act.utils import load_data
from act.act.imitate_episodes import make_policy

# 
dataset_dir = "/home/park/ros2/tensorRT/meloha_box_picking_data"
ckpt_path = "/home/park/ros2/act/ckpt/policy_best.ckpt"
camera_names = ['cam_head', 'cam_left_wrist','cam_right_wrist']
batch_size_train = 1
batch_size_val = 1
chunk_size = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# policy_config
state_dim = 6
lr_backbone = 1e-5
backbone = 'resnet18'
enc_layers = 4
dec_layers = 7
nheads = 8
policy_config = {
                'lr': 1e-5,
                'num_queries': chunk_size, # ! chunk 30으로 변경. 예전에 하던 파라미터사용할려고 60으로 맞춰놓은거임.
                'kl_weight': 10,
                'hidden_dim': 512,
                'dim_feedforward': 3200,
                'lr_backbone': lr_backbone,
                'backbone': backbone,
                'enc_layers': enc_layers,
                'dec_layers': dec_layers,
                'nheads': nheads,
                'camera_names': camera_names,
                }

policy = make_policy("ACT", policy_config)
loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
print(loading_status)
policy.cuda()
policy.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

# Load TensorRT engine from ENGINE_PATH.
ENGINE_PATH = "/home/park/ros2/tensorRT/act_origin.engine"
engine = poly_trt.engine_from_bytes(open(ENGINE_PATH, "rb").read())

train_dataloader, val_dataloader, stats, _ = load_data(
    dataset_dir,
    camera_names,
    batch_size_train,
    batch_size_val,
)

total_infer_time = 0.0
all_l1_loss = []
num_batches = 0
with poly_trt.TrtRunner(engine) as runner:
    for batch_idx, data in enumerate(val_dataloader):
        # Get inference time using 'time' module.
        image, qpos, actions, is_pad = data
        # image = normalize(image)
        before_infer = time.time()
        output_dict = runner.infer({"qpos" : qpos, "image" : image})
        pytorch_output = policy(qpos.cuda(), image.cuda())
        total_infer_time += time.time() - before_infer
        actions_hat = output_dict["action"] # shape : (1, 60, 6)

        # diff = actions_hat - pytorch_output.cpu()
        # print(diff.mean())

        # calcuate l1 error (kl divergence loss는 고려하지 않음)
        actions = actions[:, :chunk_size, :]
        is_pad = is_pad[:, :chunk_size]

        all_l1 = F.l1_loss(actions, actions_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        all_l1_loss.append(l1)
        num_batches += 1

# 평균 L1 loss와 평균 추론 시간 계산
mean_l1_loss = torch.stack(all_l1_loss).mean().item()
mean_infer_time = total_infer_time / num_batches if num_batches > 0 else 0.0

print(f"평균 L1 Loss: {mean_l1_loss:.6f}")
print(f"평균 추론 시간: {mean_infer_time*1000:.2f} ms (배치당)")


