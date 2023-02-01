import sys
import os
sys.path.append(os.path.abspath('.。/lib'))
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from runpy import run_path
from skimage import img_as_ubyte
import cv2
# from pdb import set_trace as stx
import numpy as np
from modules.paths import models_path
restormer_model = None
restormer_tile = 200
restormer_tile_overlap = 32
restormer_task = 'Motion_Deblurring' 
"""['Motion_Deblurring',
    'Single_Image_Defocus_Deblurring',
    'Deraining',
    'Real_Denoising',
    'Gaussian_Gray_Denoising',
    'Gaussian_Color_Denoising']"""

def convert_gray_img(picture):
    return np.expand_dims(cv2.cvtColor(picture,cv2.COLOR_RGB2GRAY), axis=2)

def convert_color_img(picture):
    return np.array(cv2.cvtColor(np.squeeze(picture),cv2.COLOR_GRAY2RGB))

gfpgan_constructor = None

def get_weights_and_parameters(task, parameters):
    model_path = models_path + 'restormer/'
    if task == 'Motion_Deblurring':
        weights = os.path.join(model_path, 'motion_deblurring.pth')
    elif task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join(model_path, 'single_image_defocus_deblurring.pth')
    elif task == 'Deraining':
        weights = os.path.join(model_path, 'deraining.pth')
    elif task == 'Real_Denoising':
        weights = os.path.join(model_path, 'real_denoising.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    elif task == 'Gaussian_Color_Denoising':
        weights = os.path.join(model_path, 'gaussian_color_denoising_blind.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    elif task == 'Gaussian_Gray_Denoising':
        weights = os.path.join(model_path, 'gaussian_gray_denoising_blind.pth')
        parameters['inp_channels'] =  1
        parameters['out_channels'] =  1
        parameters['LayerNorm_type'] =  'BiasFree'
    return weights, parameters

def setup_model_restormer():
    global restormer_task
    global restormer_model
    if restormer_model is None:
    # Get model weights and parameters
        parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
        weights, parameters = get_weights_and_parameters(restormer_task, parameters)

        load_arch = run_path(os.path.join('lib','basicsr_restormer', 'models', 'archs', 'restormer_arch.py'))
        restormer_model = load_arch['Restormer'](**parameters)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        restormer_model.to(device)

        checkpoint = torch.load(weights)
        restormer_model.load_state_dict(checkpoint['params'])
        restormer_model.eval()

        print(f"\n ==> Running {restormer_task} with weights {weights}\n ")

def run_restormer(images):
    global restormer_task, restormer_model
    global restormer_tile, restormer_tile_overlap
    img_multiple_of = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():

        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        if restormer_task == 'Gaussian_Gray_Denoising':
            img = np.array()
            for file_ in images.numpy():
                temp = convert_gray_img(file_)
                img = np.append(img, temp)
        else:
            img = images
        input_ = img.float().div(255.).permute(0,3,1,2).to(device)

        # Pad the input if not_multiple_of 8
        height,width = input_.shape[2], input_.shape[3]
        H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        if restormer_tile is None:
            ## Testing on the original resolution image
            restored = restormer_model(input_)
        else:
            # test the image tile by tile
            b, c, h, w = input_.shape
            tile = min(restormer_tile, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = restormer_tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h, w).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = restormer_model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                    W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
            restored = E.div_(W)

            restored = torch.clamp(restored, 0, 1)

            # Unpad the output
            restored = restored[:,:,:height,:width]

            restored = restored.permute(0, 2, 3, 1)# .cpu().detach().numpy()
            # restored = img_as_ubyte(restored)
            restored = (restored * 255.0).to(torch.uint8)

            # stx()
            if restormer_task == 'Gaussian_Gray_Denoising':
                color_img = np.array()
                for file_ in restored:
                    converted = convert_color_img(file_)
                    color_img = color_img.append(converted)
                return color_img
            else:
                return restored
