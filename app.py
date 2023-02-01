import sys
import os
sys.path.append(os.path.abspath('./lib'))
import modules.face_restoration
import modules.gfpgan_model as gfpgan
from modules.gfpgan_model import gfpgan_fix_faces
from modules import modelloader
from modules.codec import EncFFMPEG, EncTorchVis
import modules.shared as shared
from modules import esrgan_model
import gc
import math
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import center_crop, to_tensor
# import os
from modules.restormer_model import setup_model_restormer, run_restormer
from modules import restormer_model
import gc
from modules.shared import outputpath
# from PIL import Image


gfpgan_visibility =1.0
mode = 0
upscaler_idx_map = {
    'SwinIR_4x': 'SwinIR_4x',
    'Real-ESRGAN_4x':'RealESRGAN_4x',
    "ESRGAN_4x":'ESRGAN_4x',
    'Nearest': 'Nearest',
    'None': 'None',
    'Lanczos': 'Lanczos'
}

upscaler_mod_map = {
    'Real-ESRGAN_4x':[
        # "realesrgan_x2plus.onnx",
        # "realesrganv2-animevideo-xsx2.onnx",
        # "realesrganv2-animevideo-xsx4.onnx",
        "realesrgan-x4plus.onnx",
        "realesrgan-x4plus_anime_6B.onnx"
    ],
    'SwinIR_4x':["SwinIR_4x.pth"], 
    'ESRGAN_4x':["ESRGAN_4x.pth"]
}

upscaling_resize_multiplier = 4
container_constructor = EncFFMPEG
container = None

modelloader.cleanup_models()
gfpgan.setup_model(shared.gfpgan_models_path)
shared.face_restorers.append(modules.face_restoration.FaceRestoration())
modelloader.load_upscalers()


def upscale(image, scaler_index, resize, mode, resize_w, resize_h, crop):
    upscaler = shared.sd_upscalers[scaler_index]
    res = upscaler.scaler.upscale(image, resize, upscaler.data_path)
    if mode == 1 and crop:
        cropped = Image.new("RGB", (resize_w, resize_h))
        cropped.paste(res, box=(resize_w // 2 - res.width // 2, resize_h // 2 - res.height // 2))
        res = cropped
    return res

def run_gfpgan(image):
    restored_img = gfpgan_fix_faces(np.array(image, dtype=np.uint8))
    res = Image.fromarray(restored_img)

    if gfpgan_visibility < 1.0:
        res = Image.blend(image, res, gfpgan_visibility)
    return res

# This function is taken from pytorchvideo!
def uniform_temporal_subsample(x: torch.Tensor, num_samples: int, temporal_dim: int = -3) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.
    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.
    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.
    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    
    if type(x) == np.ndarray:
        x = x.transpose(3,0,1,2)
        x=torch.from_numpy(x)
    else:
        x = x.permute(3,0,1,2)
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices)

def update_upscaler_mod(upscaler_mod):
    if "realesrgan" in upscaler_mod:
        shared.RealESRModel = upscaler_mod

# This function is taken from pytorchvideo!
def short_side_scale(
    x: torch.Tensor,
    size: int,
    interpolation: str = "bilinear",
) -> torch.Tensor:
    """
    Determines the shorter spatial dim of the video (i.e. width or height) and scales
    it to the given size. To maintain aspect ratio, the longer side is then scaled
    accordingly.
    Args:
        x (torch.Tensor): A video tensor of shape (C, T, H, W) and type torch.float32.
        size (int): The size the shorter side is scaled to.
        interpolation (str): Algorithm used for upsampling,
            options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
    Returns:
        An x-like Tensor with scaled spatial dims.
    """
    assert len(x.shape) == 4
    assert x.dtype == torch.float32
    c, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))

    return torch.nn.functional.interpolate(x, size=(new_h, new_w), mode=interpolation, align_corners=False)

def get_upscaler_name(upscaler):
    for idx in range(len(shared.sd_upscalers)):
        if shared.sd_upscalers[idx].name == upscaler_idx_map[upscaler]:
            return idx
        if 'SwinIR' in shared.sd_upscalers[idx].name and 'SwinIR' in upscaler_idx_map[upscaler]:
            return idx

def inference_step(vid, duration, batch_size, out_fps, enable_GFPGAN, upscaler, restormer_task):
    video_arr, audio_arr = vid.decode(duration)
    x = uniform_temporal_subsample(video_arr, duration * out_fps)
    x = x.permute(1, 2, 3, 0)
    dest_w = x.shape[2] * upscaling_resize_multiplier
    dest_h = x.shape[1] * upscaling_resize_multiplier
    buffer = np.empty((x.shape[0], dest_h, dest_w, x.shape[3]),dtype=np.uint8) # [b, h, w, c]

    with torch.no_grad():
        ###########################################################
        for index in range(0, x.shape[0], batch_size):
            output = x[index:index+batch_size]
            if restormer_task != 'Disabled':
                output = run_restormer(output)
                # Image.fromarray(output[0].cpu().numpy()).save('a.jpg')
            buffer[index:index+batch_size] = upscale(output, get_upscaler_name(upscaler), upscaling_resize_multiplier, mode, 512, 512, crop=False)
            
            if enable_GFPGAN:
                for i in range(buffer[index:index+batch_size].shape[0]):
                    buffer[index:index+batch_size][i] = run_gfpgan(buffer[index:index+batch_size][i])

    return buffer, audio_arr



def predict_fn(filepath, start_sec, end_sec, batch_size, out_fps=-1, enable_GFPGAN=True, upscaler='SwinIR_4x', tiling=True, codec='FFMPEG', tiling_size=0, tiling_overlap=0, restormer_task='Disabled',upscaler_mod='', restormer_tiling_size=200, restormer_tiling_overlap=32):
    global container_constructor, container
    
    frame_len = 1
    name = os.path.splitext(os.path.basename(filepath))[0]
    
    update_upscaler_mod(upscaler_mod)
    if tiling:
        esrgan_model.ESRGAN_tile = tiling_size
        esrgan_model.ESRGAN_tile_overlap = tiling_overlap
    else:
        esrgan_model.ESRGAN_tile = 0
        esrgan_model.ESRGAN_tile_overlap = 0

    # del shared.face_restorers
    
    if restormer_task != 'Disabled':
        setup_model_restormer()
        restormer_model.restormer_tile = restormer_tiling_size
        restormer_model.restormer_tile_overlap = restormer_tiling_overlap

    container = container_constructor(filepath, outputpath+name+'.mp4', width=-1, height=-1, out_fps=out_fps, 
                                        start_time=start_sec, duration=end_sec - start_sec)
    if out_fps == -1:
        out_fps = container.video_fps()

    for b in  range(0, end_sec - start_sec, shared.MAX_DURATION_BATCH):
        b_start = b + start_sec
        clip_duration = shared.MAX_DURATION_BATCH if b_start+shared.MAX_DURATION_BATCH<=end_sec else end_sec - b_start
        # container.reset(b_start, clip_duration)
        msg = "🎠 Clip {}s - {}s".format(b_start, b_start+clip_duration)
        print(msg)
        for i in range(0, shared.MAX_DURATION_BATCH if b_start+shared.MAX_DURATION_BATCH<=end_sec else end_sec - b_start, frame_len):
            if b_start + i > container.total_duration:
                break
            print(f"🖼️ Processing step {i + 1}/{shared.MAX_DURATION_BATCH if b_start+shared.MAX_DURATION_BATCH<=end_sec else end_sec - b_start}...")
            video, audio = inference_step(vid=container, duration=frame_len, batch_size=batch_size, out_fps=out_fps, enable_GFPGAN=enable_GFPGAN, upscaler=upscaler, restormer_task=restormer_task)
            
            if i == 0:
                video_all = np.zeros([out_fps*clip_duration, video.shape[1], video.shape[2], video.shape[3]], np.uint8)
                audio_all = audio
                
            video_all[i*out_fps: (i+1)*out_fps] = video
            if codec != 'FFMPEG':
                audio_all = np.hstack((audio_all, audio))
            
        print(f"💾 Writing output video...")
        if not os.path.exists(outputpath):
            os.mkdir(outputpath)
        container.remux(video_all, audio_all)
        
        del audio_all
        del video_all
        # gc.collect()
    container.close()
    print(f"✅ Done!")
    return outputpath+name+'.mp4'

article = """
<p style='text-align: center'>
    <a href='https://github.com/dr413677671/SuperFast-Video-SR' target='_blank'>Github Repo Pytorch</a>
</p>
"""
def on_scaler(upscaler):
    if upscaler == 'ESRGAN_4x':
        return [gr.Checkbox.update(interactive=True), gr.Dropdown.update(choices=upscaler_mod_map[upscaler], value=upscaler_mod_map[upscaler][0])]
    else:
        return [gr.Checkbox.update(interactive=False), gr.Dropdown.update(choices=upscaler_mod_map[upscaler], value=upscaler_mod_map[upscaler][0])]

def on_enable_tiling(option):
    if option:
        return [gr.Slider.update(visible=True), gr.Slider.update(visible=True)]
    else:
        return [gr.Slider.update(visible=False), gr.Slider.update(visible=False)]

def on_codec(codec):
    global container_constructor
    
    if codec == 'FFMPEG':
        container_constructor = EncFFMPEG
    elif codec == 'TorchVision':
        container_constructor = EncTorchVis

def on_size(size):
    esrgan_model.ESRGAN_tile = size

def on_overlap(overlap):
    esrgan_model.ESRGAN_tile_overlap = overlap

def on_restormer(restormer_type):
    if restormer_type != "Disabled":
        return [ gr.Slider.update(visible=True), gr.Slider.update(visible=True)]
    else:
        return [gr.Slider.update(visible=False), gr.Slider.update(visible=False)]


def on_video_change(filepath, start_sec, out_fps):
    global container
    if filepath is None:
        return 
    name = os.path.splitext(os.path.basename(filepath))[0]
    container = container_constructor(filepath, outputpath+name+'.mp4', width=-1, height=-1, out_fps=out_fps, start_time=0, duration=1)
    max_duration = container.get_total_duration()
    container.get_stats()
    stats = [ (key, value) for key,value in container.get_stats()['video'].items()]
    stats = ""
    stats += '**video:**  \n    '
    for key,value in container.get_stats()['video'].items():
        stats += "   {}: {}".format(key, value)
    stats += '  \n  **audio:**  \n    '
    for key,value in container.get_stats()['audio'].items():
        stats += " {}:{}".format(key, value)
    del container
    return [gr.Slider.update(value=start_sec if start_sec<max_duration else 0, maximum=max_duration), 
            gr.Slider.update(value=round(max_duration) , maximum=max_duration),\
            stats
            ]


def main_tab():
    with gr.Row():
        with gr.Column():
            inputs_video = gr.Video(source="upload", interactive=True)
            stats = gr.Markdown()

            with gr.Group() as meta:
                start_sec = gr.Slider(minimum=0, maximum=3000, step=1, value=0, label="start_sec", interactive=True)
                end_sec = gr.Slider(minimum=1, maximum=3000, step=1, value=6, label="end_sec", interactive=True)
            with gr.Group() as video_setup:
                batch_size = gr.Slider(minimum=1, maximum=24, step=1, value=8, label="batch_size", interactive=True)
                out_fps = gr.Slider(minimum=-1, maximum=30, step=1, value=24, label="output_fps (original fps: -1)", interactive=True)
            
        
            with gr.Row():
                enable_GFPGAN = gr.Checkbox(value=True, label="Enable GFPGAN (face restoration)", interactive=True)
                enable_tiling = gr.Checkbox(value=False, label="Enable Tiling (ESRGAN only)", interactive=False)
            with gr.Row():
                restormer_task = gr.Dropdown(value='Disabled', choices=['Disabled','Motion_Deblurring','Single_Image_Defocus_Deblurring','Deraining','Real_Denoising','Gaussian_Gray_Denoising','Gaussian_Color_Denoising'], label='Restormer', interactive=True)
            with gr.Group() as restormer_tile:
                restormer_tiling_size = gr.Slider(minimum=-1, maximum=244, value=200, label="Restormer Tile Size (-1: disabled)", visible=False)
                restormer_tiling_overlap = gr.Slider(minimum=0, maximum=32, value=32, label="Restormer Tile Overlap", visible=False)
            with gr.Group() as tiling_setup:
                tiling_size = gr.Slider(minimum=0, maximum=244, value=198, label="Tile Size", visible=False)
                tiling_overlap = gr.Slider(minimum=0, maximum=32, value=8, label="Tile Overlap", visible=False)
            with gr.Group() as upscaler_setup:
                upscaler = gr.Radio(choices=['SwinIR_4x','Real-ESRGAN_4x', 'ESRGAN_4x'], label="Upscaler",value='SwinIR_4x')
                upscaler_mod = gr.Dropdown(choices=["SwinIR_4x.pth"], label="upscaler_mod",value="SwinIR_4x.pth",interactive=True)
                
            with gr.Group() as codec_setup:
                codec = gr.Radio(choices=['FFMPEG', 'TorchVision'], value="FFMPEG", label="Codec_wrapper")

            with gr.Group() as control:
                with gr.Row():
                    clear = gr.Button("Clear")
                    submit = gr.Button("Submit")
    
        outputs_video = gr.Video(interactive=False)
    gr.Markdown("* [github](https://github.com/dr413677671/super-video-super-resolution)")
    examples = gr.Examples([
        ['./assets/examples/driving.mp4', 0, 6],
        ['./assets/examples/bella_poarch.mp4', 0, 8]
    ], inputs=[inputs_video, start_sec, end_sec], outputs=[inputs_video, start_sec, end_sec])

    upscaler.change(fn=on_scaler, inputs=[upscaler], outputs=[enable_tiling, upscaler_mod])
    enable_tiling.change(fn=on_enable_tiling, inputs=[enable_tiling], outputs=[tiling_size, tiling_overlap])
    codec.change(fn=on_codec, inputs=[codec], outputs=[])
    submit.click(fn=predict_fn, inputs=[inputs_video, start_sec, end_sec, batch_size, out_fps, enable_GFPGAN, upscaler, enable_tiling, codec, tiling_size, tiling_overlap, restormer_task, upscaler_mod, restormer_tiling_size, restormer_tiling_overlap], outputs=[outputs_video])
    inputs_video.change(fn=on_video_change, inputs=[inputs_video, start_sec, out_fps], outputs=[start_sec, end_sec, stats])
    restormer_task.change(fn=on_restormer, inputs=[restormer_task], outputs=[restormer_tiling_size, restormer_tiling_overlap])

with gr.Blocks(title="Superfast Video SR") as app:
    gr.Label("Superfast Video SR")
    with gr.Tab(label="Super Resolution"):
        main_tab()


if __name__ == '__main__':
    app.launch()