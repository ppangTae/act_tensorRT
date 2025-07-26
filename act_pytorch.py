
import torch
from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
act_dir = parent_dir / "act" / "act"
sys.path.append(str(act_dir))

from imitate_episodes import make_policy

# ===== 0. set the variables =====
dataset_dir = parent_dir / "data" / "meloha_box_picking"
ckpt_dir = parent_dir / "act" / "ckpt"
ckpt_path = parent_dir / "act" / "ckpt" / "policy_best.ckpt"
camera_names = ['cam_head', 'cam_left_wrist','cam_right_wrist']
batch_size_train = 1
batch_size_val = 1
chunk_size = 30 # !
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# policy_config
state_dim = 6
lr_backbone = 1e-5
backbone = 'resnet18'
enc_layers = 3 # ! 
dec_layers = 6 # ! 
nheads = 8
policy_config = {
                'lr': 2e-6,
                'num_queries': chunk_size,
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

def make_act_pytorch_model():
    policy = make_policy("ACT", policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return policy