import torch
import os
import random
from PIL import Image
from torchvision import transforms
import utils
from torch.utils import data
import json
# import resource

# Nullify too many open files error
# _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

class VideoFramesDataset(data.Dataset):
    """Some Information about VideoFramesDataset"""
    def __init__(self, root_dir, frame_num=0, transform=None):
        super(VideoFramesDataset, self).__init__()

        self.samples = []
        self.transform = transform
        self.frame_num = frame_num

        self.cls_lst = os.listdir(root_dir)  # 此处跟目录结构有关参看readme.md中的目录结构
        self.num_classes = len(self.cls_lst)  # 总的行为的类别数量
        # Import data in root_dir, each subfolder corresponds to a class label
        if not os.path.exists('vidoe2classid.txt'):
            #将视频数据进行处理（video_path,classid）并存入文件中
            print("=====begin process videos=====")
            for i in range(self.num_classes): #(video_path,class_id)存储在self.samples数组中
                cls_dir = os.path.join(root_dir, self.cls_lst[i])
                for video in os.listdir(cls_dir):
                    video_path = os.path.join(cls_dir, video)
                    if len(os.listdir(video_path)) > self.frame_num:
                        self.samples.append((os.path.join(cls_dir, video), i) )
            # 缓存到文件中
            with open('vidoe2classid.txt', 'w') as f:
                content = json.dumps(self.samples)
                f.write(content)
        else:
            f = open('vidoe2classid.txt','r')
            content = f.read()
            self.samples = json.loads(content)


    def __getitem__(self, index):
        sample = self.samples[index]
        frame_paths = [os.path.join(sample[0], f) for f in os.listdir(sample[0])] #一段视频中的所有帧

        if self.frame_num > 1:
            # Get a random sequence of frames
            frame_index = random.randrange(0, len(frame_paths) - self.frame_num)
            frame_paths = frame_paths[frame_index:frame_index + self.frame_num]

        frames = [Image.open(f) for f in frame_paths]

        if self.transform is not None:
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)
        return frames, sample[1]

    def __len__(self):
        return len(self.samples)
