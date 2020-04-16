import torch
import os
import random
from PIL import Image
from torchvision import transforms
import utils
from torch.utils import data
import json
import  numpy as np
# import resource

# Nullify too many open files error
# _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

class VideoFramesDataset(data.Dataset):
    """Some Information about VideoFramesDataset"""
    def __init__(self, root_dir, frame_num=16, transform=None,num_segments=1,split=''):
        super(VideoFramesDataset, self).__init__()

        self.samples = []
        self.transform = transform
        self.frame_num = frame_num
        self.num_segments = num_segments
        self.split = split

        self.cls_lst = os.listdir(root_dir)  # 此处跟目录结构有关参看readme.md中的目录结构
        self.num_classes = len(self.cls_lst)  # 总的行为的类别数量

        with open(split,'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().replace('\n','')
                video_name,total_num,classid = line.split(' ')
                video_class = video_name.split("_")[1]
                self.samples.append((root_dir+video_class+'/'+video_name,int(classid[0])))
                

        # Import data in root_dir, each subfolder corresponds to a class label
        # if not os.path.exists('vidoe2classid.txt'):
        #     #将视频数据进行处理（video_path,classid）并存入文件中
        #     print("=====begin process videos=====")
        #     for i in range(self.num_classes): #(video_path,class_id)存储在self.samples数组中
        #         cls_dir = os.path.join(root_dir, self.cls_lst[i])
        #         for video in os.listdir(cls_dir):
        #             video_path = os.path.join(cls_dir, video)
        #             if len(os.listdir(video_path)) > self.frame_num:
        #                 self.samples.append((os.path.join(cls_dir, video), i) )
        #     # 缓存到文件中
        #     with open('vidoe2classid.txt', 'w') as f:
        #         content = json.dumps(self.samples)
        #         f.write(content)
        # else:
        #     f = open('vidoe2classid.txt','r')
        #     content = f.read()
        #     self.samples = json.loads(content)

    def get_indices(self, frames_paths):
        num_frames = len(frames_paths)
        if num_frames > self.num_segments + self.frame_num - 1:
            tick = (num_frames - self.frame_num + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _load_image(self,frame_path):
        return Image.open(frame_path).convert('RGB')

    def __getitem__(self, index):
        sample = self.samples[index]
        frame_paths = [os.path.join(sample[0], f) for f in os.listdir(sample[0])] #当前这段视频中的所有帧
        segment_indices = self.get_indices(frame_paths)

        # Get a random sequence of frames
        frames = self.get(frame_paths,segment_indices)

        frames = torch.stack(frames)
        return frames, sample[1]

    def get(self,frame_paths,indices):
        #获取一段视频的三帧
        # log(indices)
        num_frames = len(frame_paths)
        images = list()
        for seg_ind in indices:
            #分段采集
            p = int(seg_ind)
            for i in range(self.frame_num):
                #采集这一段的new_length帧图片，RGB模态的为1帧，optical flow模态的为5帧
                seg_img = self._load_image(frame_paths[p]) #读入一帧图片
                images.append(seg_img)
                if p < num_frames-1:
                    p += 1
        process_data = [self.transform(frame) for frame in images]
        return process_data

    def __len__(self):
        return len(self.samples)
