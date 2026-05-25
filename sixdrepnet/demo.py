import time
import math
import re
import sys
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
from numpy.lib.function_base import _quantile_unchecked
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from face_detection import RetinaFace
from PIL import Image

from model import SixDRepNet
import utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0], set -1 to use CPU',
                        default=0, type=int)
    parser.add_argument('--cam',
                        dest='cam_id', help='Camera device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--video',
                        dest='video_path', help='Path to a video file to use instead of a camera.',
                        default='', type=str)
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--output',
                        dest='output', help='Output video filename saved in the current directory.',
                        default='result.mp4', type=str)
    parser.add_argument('--save_viz',
                        dest='save_viz', help='Save images with pose cube.',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    if (gpu < 0):
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % gpu)
    video_path = args.video_path
    cam = args.cam_id
    output_path = os.path.join(os.getcwd(), os.path.basename(args.output))
    snapshot_path = args.snapshot
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

    print('Loading data.')

    detector = RetinaFace(gpu_id=gpu)

    # Load snapshot
    saved_state_dict = torch.load(os.path.join(
        snapshot_path), map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    model.to(device)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    input_source = video_path if video_path else cam
    cap = cv2.VideoCapture(input_source)

    # Check if the video source is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open video source: %s" % input_source)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    writer = None

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise IOError("Cannot open output video: %s" % output_path)

            faces = detector(frame)

            for box, landmarks, score in faces:

                # Print the location of each face in this image
                if score < .95:
                    continue
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max = x_max+int(0.2*bbox_height)
                y_max = y_max+int(0.2*bbox_width)

                img = frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations(img)

                img = torch.Tensor(img[None, :]).to(device)

                start = time.time()
                R_pred = model(img)
                end = time.time()
                print('Head pose estimation: %2f ms' % ((end - start)*1000.))

                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred)*180/np.pi
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                #utils.draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg, left+int(.5*(right-left)), top, size=100)
                utils.plot_pose_cube(frame,  y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5*(
                    x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)

            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    print('Saved result video to %s' % output_path)
