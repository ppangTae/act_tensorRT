import os
import sys
import h5py
import numpy as np
import time
import argparse
import torch
from PIL import Image
from polygraphy.backend import trt as poly_trt
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/home/park/ros2/act/act")
from act.act.utils import get_norm_stats

# 1. Dataset에서 representative dataset을 추린다.
# -  100개의 episode에서 총 10개의 episode를 뽑는다고 생각하고, 30hz제어, chunk size : 30, 총 30s 제어한다고 생각하면
# -  10개의 episode는 9000개의 (qpos, image)쌍이 존재한다.

dataset_dir = "/home/park/ros2/tensorRT/meloha_box_picking_data"
camera_names = ['cam_head', 'cam_left_wrist','cam_right_wrist']

# ! You should change your own data's episode id
representative_episode_ids = [0, 10, 20, 30, 40]

# 이 함수는 INT8 엔진에서 calibration을 통해 정확도가 얼마나 향상되는지 판단하기 위해 사용됩니다.
def random_data_generator(norm_stats):
    for _ in range(1):
        qpos = torch.randn(1, 6).numpy()
        image = torch.randn(1, 3, 3, 480, 640).numpy()

        qpos = (qpos - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
        yield {"qpos" : qpos, "image" : image}

def data_generator(norm_stats, representative_episode_ids):
    for episode_id in representative_episode_ids:
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, 'r') as root:
            qpos_data = root['/observations/qpos'][()] # shape : (episode_len, qpos)
            image_dict = dict()
            for cam_name in camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()] # shape : (episode_len, 480, 640, 3)

            # new axis for different cameras
            all_cam_images = []
            for cam_name in camera_names: 
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0) # shape : (# of camera, episode_len, 480, 640 ,3)

            # channel last
            image_data = torch.from_numpy(all_cam_images)
            qpos_data = torch.from_numpy(qpos_data)
            image_data = torch.einsum('b k h w c -> k b c h w', image_data) # shape : (episode_len, # of camera, 3, 480, 640)

            # Change tensor to numpy
            image_data = image_data.numpy()
            qpos_data = qpos_data.numpy()

            # normalize image and change dtype to float
            image_data = image_data / np.float32(255.0)
            qpos_data = (qpos_data - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]

        for qpos, image in zip(qpos_data, image_data):
            # Dict key must be the same as ONNX input name
            qpos = np.expand_dims(qpos, axis=0)
            image = np.expand_dims(image, axis=0)
            yield {"qpos" : qpos, "image" : image}


def main(args):
    
    onnx_model_path = args["onnx_model_path"]
    engine_path = args["engine_path"]
    calibration = args["calibration"]
    fp16 = args["fp16"]
    fp32 = args["fp32"]
    tf32 = args["tf32"]
    dataest_dir = args["dataset_dir"]
    episode_ids = args["episode_ids"]

    norm_stats = get_norm_stats(dataset_dir, 41)

    builder, network, parser = poly_trt.network_from_onnx_path(path=onnx_model_path)

    if fp32:
        builder_config = poly_trt.create_config(builder=builder,
                                                network=network)
    else:
        if calibration:
            calibrator = poly_trt.Calibrator(data_loader=data_generator(norm_stats, episode_ids),
                                             cache="act.cache")
            print("calibration을 진행합니다.")
        else:
            calibrator = poly_trt.Calibrator(data_loader=random_data_generator(norm_stats))
            print("calibration을 진행하지 않습니다.")

        # Each type flag must be set to true.
        builder_config = poly_trt.create_config(builder=builder, network=network,
                                                int8=True, fp16=fp16, tf32=tf32,
                                                calibrator=calibrator)

    engine = poly_trt.engine_from_network(network=(builder, network, parser), 
                                          config=builder_config)

    # TensorRT engine will be saved to ENGINE_PATH.
    poly_trt.save_engine(engine, engine_path)

    # Load serialized engine using 'open'.
    engine = poly_trt.engine_from_bytes(open(engine_path, "rb").read())
    print("엔진이 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", action="store", type=str,
                        default="/home/park/ros2/tensorRT/act.onnx",
                        help="입력 ONNX 모델 파일 경로")
    parser.add_argument("--engine_path", action="store", type=str,
                        default="/home/park/ros2/tensorRT/act_no_calib_int8_fp16.engine",
                        help="TensorRT Engine path")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--dataset_dir", action="store", type=str,
                        default="/home/park/ros2/tensorRT/meloha_box_picking_data",
                        help="대표 데이터셋 디렉터리")
    parser.add_argument("--episode_ids", action="store", type=int, nargs="+",
                        default=[0, 10, 20, 30, 40],
                        help="대표 episode id 리스트 (space‑separated)")
    argument = vars(parser.parse_args())

    main(vars(parser.parse_args()))

    # 사용법
    # int8 양자화, calibraiton, int8 연산지원이 안되는 레이어에 대해서 fp16, tf32 허용 -> python3 build_engine.py --calibration --fp16 --tf32