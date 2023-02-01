import os
import argparse
import modules.devices as devices
import torch

script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
models_path = os.path.join(script_path, "models")

device = None
sd_upscalers = []

no_half = True
cpu = torch.device("cpu")
device = device_interrogate = device_gfpgan = device_swinir = device_esrgan = device_scunet = device_codeformer = None
dtype = torch.float16
dtype_vae = torch.float16

devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_swinir, devices.device_esrgan, devices.device_scunet, devices.device_codeformer = \
(devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'swinir', 'esrgan', 'scunet', 'codeformer'])

gfpgan_models_path=os.path.join(models_path, 'GFPGAN')
face_restorers = []
face_restoration_unload = False

parser = argparse.ArgumentParser()
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN'))
parser.add_argument("--gfpgan-model", type=str, help="GFPGAN model file name", default=None)
parser.add_argument("--unload-gfpgan", action='store_true', help="does not do anything.")
parser.add_argument("--gfpgan-models-path", type=str, help="Path to directory with GFPGAN model file(s).", default=os.path.join(models_path, 'GFPGAN'))
cmd_opts = parser.parse_args()

RealESRModel = ""
outputpath = './outputs/'
MAX_DURATION_BATCH = 200