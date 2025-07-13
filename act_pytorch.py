
import torch
from act.act.imitate_episodes import make_policy

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

def make_act_pytorch_model():
    policy = make_policy("ACT", policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return policy