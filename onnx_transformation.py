import os
import sys
import torch
import argparse
from pprint import pprint


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from act.act.imitate_episodes import make_policy

def main(args):

    save_dir = "/home/park/ros2/tensorRT"
    ckpt_path = "/home/park/ros2/act/ckpt/policy_best.ckpt"
    camera_names = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fixed parameters
    state_dim = 6
    lr_backbone = 1e-5
    backbone = 'resnet18'
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {
                    'lr': 1e-5,
                    'num_queries': 60, # ! chunk 30으로 변경. 예전에 하던 파라미터사용할려고 60으로 맞춰놓은거임.
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

    batch_size = 1
    state_dim = 6
    num_cameras = 3
    channels = len(camera_names)
    height = 480
    width = 640

    dummy_qpos = torch.randn(batch_size, state_dim).cuda()
    dummy_image = torch.randn(batch_size, num_cameras, channels, height, width).cuda()

    # ONNX로 export
    onnx_path = os.path.join(save_dir, "policy.onnx")
    torch.onnx.export(
        policy,                                 # 변환할 모델
        (dummy_qpos, dummy_image),              # 입력 튜플
        onnx_path,                              # 저장 경로
        export_params=True,                     # 모델 파라미터 저장
        opset_version=24,                       # ONNX opset 버전
        do_constant_folding=True,               # 상수 폴딩 최적화
        input_names=['qpos', 'image'],          # 입력 이름
        output_names=['action'],                # 출력 이름 (모델에 따라 다를 수 있음)
    )
    print(f"ONNX 모델이 {onnx_path}에 저장되었습니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # for ACT
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))