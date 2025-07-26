import datetime
import os
import sys
import time
import collections

import torch
import torch.utils.data
from torch import nn
from pathlib import Path

from tqdm import tqdm

import torchvision
from torchvision import transforms


from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

parent_dir = Path(__file__).resolve().parent.parent
act_dir = parent_dir / "act" / "act"
sys.path.append(str(parent_dir))
sys.path.append(str(act_dir))

from act_pytorch import make_act_pytorch_model
from act.act.utils import load_data
from act.act.imitate_episodes import make_optimizer
from qat_utils import train_bc, eval_bc

from absl import logging
logging.set_verbosity(logging.FATAL)  # Disable logging as they are too noisy in notebook

# ===== 0. set the variables =====
dataset_dir = parent_dir / "data" / "meloha_box_picking"
ckpt_dir = parent_dir / "act" / "ckpt"
ckpt_path = parent_dir / "act" / "ckpt" / "policy_best_enc4_dec7_chunk60.ckpt" # !
camera_names = ['cam_head', 'cam_left_wrist','cam_right_wrist']
batch_size_train = 8
batch_size_val = 8
chunk_size = 60 # !
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 1. Set default QuantDescriptor to use histogram based calibration for activation =====
quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

# ===== 2. Initialize =====
from pytorch_quantization import quant_modules
quant_modules.initialize() # 이 함수를 통해 이후에 등록하는 모델의 레이어를 Q/DQ가 삽입된 레이어로 변경

# ===== 3. Create model with pretrained weight =====
policy = make_act_pytorch_model()
loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
optimizer = make_optimizer('ACT', policy)
policy.cuda()

# ===== 4. Create Data loader =====
train_dataloader, val_dataloader, stats, _ = load_data(
    str(dataset_dir),
    camera_names,
    batch_size_train,
    batch_size_val,
)

# ===== 5. Train policy ======

best_ckpt_info = train_bc(policy=policy,
                          train_dataloader=train_dataloader,
                          val_dataloader=val_dataloader,
                          num_epochs=400,
                          optimizer=optimizer
                          )

best_epoch, min_val_loss, best_state_dict = best_ckpt_info

# save best checkpoint
ckpt_path = os.path.join(ckpt_dir, 'policy_best_qat_enc4_dec7_chunk60.ckpt')
torch.save(best_state_dict, ckpt_path)
print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')
