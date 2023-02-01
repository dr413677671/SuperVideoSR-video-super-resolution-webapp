from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.io import read_video, write_video
from abc import ABC
import numpy as np
import ffmpeg
import queue

class Enc(ABC):
    def __init__(self, output, start_time, duration):
        self.start_pos = start_time
        self.output = output
        self.duration = duration
    def get_clip(self):
        pass
    def video_fps(self):
        pass
    def audio_rate(self):
        pass
    def decode_video(self):
        pass
    def decode_audio(self):
        pass
    def get_total_duration(self):
        return self.total_duration
    def get_stats(self):
        pass

# TBD
class ConsumerInterface():
    def __init__(self):
        pass

# TBD
class Consumer(ConsumerInterface):
    def __init__(self, output, start_time, duration):
        broker = Task
        _task_queue= queue.Queue()

class EncTorchVis(Enc):
    def __init__(self, path, output, width=-1, height=-1, out_fps=-1, start_time=0, duration=10):
        super().__init__(output=output, start_time=start_time, duration=duration)
        self.container = EncodedVideo.from_path(path)
        self.clip = self.container.get_clip(self.start_pos, 1)
        self.fps = self.clip['video'].shape[1]
        self.aud_rate = None if not self.container._has_audio else self.container._container.streams.audio[0].sample_rate
        self.width = self.container._container.streams.video[0].width
        self.total_duration = int(self.container._container.streams.video[0].frames / self.fps)
        self.aud_codec_name = None if not self.container._has_audio else self.container._container.streams.audio[0].codec_context.codec.name
        self.vid_codec_name = self.container._container.streams.video[0].codec_context.codec.name
        self.vid_pix_format = self.container._container.streams.video[0].codec_context.pix_fmt

        self.out_width = int(self.container._container.streams.video[0].width) if width == -1 else width
        self.out_height = int(self.container._container.streams.video[0].height) if height == -1 else height
        self.out_fps = self.fps if out_fps == -1 else out_fps

    def get_stats(self):
        return {
            'video':
            {
                'pix_format': self.vid_pix_format,
                'codec': self.vid_codec_name,
                'fps': self.fps,
                'height': self.container._container.streams.video[0].height,
                'width': self.width
            },
            'audio':
            {
                'codec': self.aud_codec_name,
                'sample_rate': self.aud_rate
            }

        }
    def reset(self, start_pos, duration):
        self.start_pos = start_pos
        self.total_duration = duration
    def add_file(self, path):
        self.container = EncodedVideo.from_path(path)
        self.clip = None
    def decode(self, duration=1):
        self.clip = self.container.get_clip(self.start_pos, self.start_pos+duration)
        self.start_pos += duration
        return self.clip['video'], np.expand_dims(self.clip['audio'], 0)
    def decode_video(self):
        return self.clip['video']
    def decode_audio(self):
        return np.expand_dims(self.clip['audio'], 0)
    def video_fps(self):
        return self.fps
    def audio_rate(self):
        return self.aud_rate
    def get_width(self):
        return self.width
    def get_height(self):
        return self.container._container.streams.video[0].height
    def remux(self, vid, aud):
        try:
            write_video(self.output, vid, fps=self.out_fps, audio_array=aud, audio_fps=self.aud_rate,audio_codec='aac')
        except:
            print("❌ Error when writing with audio...trying without audio")
            write_video(self.output, vid, fps=self.out_fps)


class EncFFMPEG(Enc):
    def __init__(self, path, output, width=-1, height=-1, out_fps=-1, start_time=0, duration=10):
        super().__init__(output=output, start_time=start_time, duration=duration)
        self.probe = ffmpeg.probe(path)
        self.video_stream = next((stream for stream in self.probe['streams'] if stream['codec_type'] == 'video'), None)
        self.aud  = ffmpeg.input(path, ss=self.start_pos, t=self.duration)
        self.width = int(self.video_stream['width'])
        self.height = int(self.video_stream['height'])

        self.path = path
        self.vid_fps = round(int(self.video_stream['r_frame_rate'].split('/')[0])/int(self.video_stream['r_frame_rate'].split('/')[1]))
        self.total_duration = int(float(self.video_stream['duration']))
        self.vid_codec_name = self.video_stream['codec_name']
        self.vid_bitrate = int(self.video_stream['bit_rate'])
        self.pix_format = self.video_stream['pix_fmt']

        self.aud_stream = next(s for s in self.probe['streams'] if s['codec_type'] == 'audio')
        self.aud_rate = int(self.aud_stream['sample_rate'])
        
        self.aud_channels = int(self.aud_stream['channels'])
        self.aud_codec_name = self.aud_stream['codec_name']
        self.aud_bitrate = int(self.aud_stream['bit_rate'])
        try:
            self.aud_channel_layout = self.aud_stream['channel_layout']
        except:
            self.aud_channel_layout = "N/A"
        self.aud_sample_format = self.aud_stream['sample_fmt']


        self.out_width = int(self.video_stream['width']) if width == -1 else width
        self.out_height = int(self.video_stream['height']) if height == -1 else height
        self.out_fps = self.vid_fps if out_fps == -1 else out_fps
        self.inSink = (
            ffmpeg
            .input(path, ss=self.start_pos, t=self.duration)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=self.vid_fps)
            .run_async(pipe_stdout=True)
        )

        self.outSink = None

    def reset(self, start_pos, duration):
        self.inSink.stdout.close()
        # self.inSink.close()
        # del self.inSink
        self.start_pos = start_pos
        self.duration = duration
        self.inSink = (
            ffmpeg
            .input(self.path, ss=self.start_pos, t=self.duration)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=self.vid_fps)
            .run_async(pipe_stdout=True)
        )
    def get_stats(self):
        return { 
            'video':
            {
                'height': self.height,
                'width': self.width,
                'fps': self.vid_fps,
                'duration (s)': self.total_duration,
                'codec': self.vid_codec_name,
                'bitrate':self.vid_bitrate,
                'pix_format':self.pix_format
            },      
            'audio':
            {
                'sample_rate': self.aud_rate,
                'channels': self.aud_channels,
                'codec': self.aud_codec_name,
                'bitrate': self.aud_bitrate,
                'channel_layout': self.aud_channel_layout,
                'sample_format': self.aud_sample_format
            }
        }

    def decode(self, duration=1):
        vid_bytes = self.inSink.stdout.read(duration * self.vid_fps * self.width * self.height * 3)
        vid_frame = (
            np
            .frombuffer(vid_bytes, np.uint8)
            .reshape([self.vid_fps, self.height, self.width, 3])
        )
        return vid_frame, None

    def init_outSink(self, vid):
        self.out_height = vid.shape[1]
        self.out_width = vid.shape[2]
        self.vid_out = ffmpeg.input('pipe:', format='rawvideo', codec="rawvideo", pix_fmt='rgb24', s='{}x{}'.format(self.out_width,self.out_height), r=str(self.out_fps))# .format(width, height)
        self.outSink  = (
            ffmpeg
            .concat(self.vid_out, self.aud, self.aud, v=1, a=2)
            .output(self.output, pix_fmt='yuv420p',  vcodec='libx264', r=str(self.out_fps))
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        self.outSink.args[self.outSink.args.index('-s')+1] = "{}x{}".format(self.out_width, self.out_height)

    def remux(self, vid, aud):
        # vid = vid.transpose(0,1,2,3)
        
        if self.outSink is None:
            self.init_outSink(vid)
        # self.outSink.stdin.open()
        for frame in np.asarray(vid):
            self.outSink.stdin.write(
                (frame.astype(np.uint8).tobytes())
            )
        # self.outSink.stdin.close()
        # self.outSink.wait()
    def video_fps(self):
        return self.vid_fps
    def audio_rate(self):
        return self.aud_rate
    def video_fps(self):
        return self.vid_fps
    def get_width(self):
        return self.width
    def get_height(self):
        return self.height
    def close(self):
        self.outSink.stdin.close()
        self.outSink.wait()