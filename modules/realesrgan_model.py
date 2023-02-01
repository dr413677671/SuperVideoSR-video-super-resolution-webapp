import sys
import os
sys.path.append(os.path.abspath('.。/lib'))
import math
import onnxruntime as rt
import cv2
import numpy as np
import numpy as np
import torch
from modules.upscaler import Upscaler, UpscalerData
from basicsr.utils.download_util import load_file_from_url
from modules import modelloader
from modules import shared

class UpscalerRealESRGAN(Upscaler):
    def __init__(self, dirname):
        self.name = "RealESRGAN"
        self.model_url = "./models/RealESRGAN/" + shared.RealESRModel
        self.model_name = "RealESRGAN_4x"
        self.scalers = []
        self.user_path = dirname
        super().__init__()
        model_paths = [self.model_url]
        scalers = []
        if len(model_paths) == 0:
            scaler_data = UpscalerData(self.model_name, self.model_url, self, 4)
            scalers.append(scaler_data)
        self.sess = None
        name = self.model_name
        scaler_data = UpscalerData(name, self.model_url, self, 4)
        self.scalers.append(scaler_data)

    def load_model(self, path: str):
        if not 'onnx' in path:
            path = path + shared.RealESRModel
        if "http" in path:
            filename = load_file_from_url(url=self.model_url, model_dir=self.model_path,
                                          file_name="%s.onnx" % self.model_name,
                                          progress=True)
        if 'CUDAExecutionProvider' in rt.get_available_providers():
            providers = ['CUDAExecutionProvider']
        elif 'TensorrtExecutionProvider' in rt.get_available_providers():
            providers = ['TensorrtExecutionProvider']
        else:
            providers = ["DmlExecutionProvider"]
        self.sess = rt.InferenceSession(path, providers=providers)# provider_options={"deviceId": "1"}
        return self.sess

    def nearest_of_value(self, x, base):  
        return math.ceil(x/base)*base

    def pad(self, img):
        pad_w = self.nearest_of_value(img.shape[0], 64) - img.shape[0]
        pad_h = self.nearest_of_value(img.shape[1], 64) - img.shape[1]
        return (pad_w, pad_h), cv2.copyMakeBorder(img, 0,  pad_w, 0, pad_h, cv2.BORDER_REFLECT)
    
    def crop_center(self, img,cropx,cropy):
        y,x,_ = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return self.crop(img, startx, starty, cropx, cropy)

    def crop(self, img, startx, starty, cropx, cropy):
        return img[starty:starty+cropy,startx:startx+cropx]

    def do_upscale(self, img, selected_model):
        if self.sess is None:
            self.sess = self.load_model(selected_model)
        img = img.permute(0,1,3,2)
        img = img.cpu().numpy()

        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name
        
        out_mats = self.sess.run([output_name], {input_name: img})[0]
        out_mats = out_mats.clip(0, 1)

        img = torch.tensor(out_mats)
        img = img.permute(0,1,3,2)
        return img

   