import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import lmdb
import numpy as np
import time


def resize_and_convert(img, size, resample):
    if(img.size[0] != size):
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img


def image_convert_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format='png')
    return buffer.getvalue()

# 修改后的函数，从两个文件夹读取对应的图片
def process_image_pair(gt_file, lr_file, sizes=(256, 256), resample=Image.BICUBIC, lmdb_save=False):
    # 读取gt图片作为hr_img
    gt_img = Image.open(gt_file)
    gt_img = gt_img.convert('RGB')
    hr_img = resize_and_convert(gt_img, sizes[1], resample)
    
    # 读取lr图片作为lr_img
    lr_img = Image.open(lr_file)
    lr_img = lr_img.convert('RGB')
    lr_img = resize_and_convert(lr_img, sizes[0], resample)
    
    # sr_img就是把lr_img上采样到hr_img的尺寸
    sr_img = resize_and_convert(lr_img, sizes[1], resample)

    if lmdb_save:
        lr_img = image_convert_bytes(lr_img)
        hr_img = image_convert_bytes(hr_img)
        sr_img = image_convert_bytes(sr_img)

    return [lr_img, hr_img, sr_img]

def process_worker(gt_file, lr_file, sizes, resample, lmdb_save=False):
    out = process_image_pair(
        gt_file, lr_file, sizes=sizes, resample=resample, lmdb_save=lmdb_save)
    
    # 直接返回处理结果，不返回文件名
    return out

class WorkingContext():
    def __init__(self, process_fn, lmdb_save, out_path, env, sizes):
        self.process_fn = process_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
        self.sizes = sizes

        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value
    
    def get_next_index(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

def prepare_process_worker(wctx, file_pairs):
    for gt_file, lr_file in file_pairs:
        imgs = wctx.process_fn(gt_file, lr_file)
        lr_img, hr_img, sr_img = imgs
        # 获取下一个序号
        index = wctx.get_next_index()
        index_str = str(index).zfill(6)
        
        if not wctx.lmdb_save:
            lr_img.save(
                '{}/lr_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], index_str))
            hr_img.save(
                '{}/hr_{}/{}.png'.format(wctx.out_path, wctx.sizes[1], index_str))
            sr_img.save(
                '{}/sr_{}_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], wctx.sizes[1], index_str))
        else:
            with wctx.env.begin(write=True) as txn:
                txn.put('lr_{}_{}'.format(
                    wctx.sizes[0], index_str).encode('utf-8'), lr_img)
                txn.put('hr_{}_{}'.format(
                    wctx.sizes[1], index_str).encode('utf-8'), hr_img)
                txn.put('sr_{}_{}_{}'.format(
                    wctx.sizes[0], wctx.sizes[1], index_str).encode('utf-8'), sr_img)
        curr_total = wctx.value()
        if wctx.lmdb_save:
            with wctx.env.begin(write=True) as txn:
                txn.put('length'.encode('utf-8'), str(curr_total).encode('utf-8'))

def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True

def prepare(gt_path, lr_path, out_path, n_worker, sizes=(256, 256), resample=Image.BICUBIC, lmdb_save=False):
    # 获取gt文件夹下的所有图片文件
    gt_files = sorted([p for p in Path(gt_path).glob(f'**/*') if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    lr_files = sorted([p for p in Path(lr_path).glob(f'**/*') if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    
    # 确保两个文件夹下的文件数量相同且文件名对应
    if len(gt_files) != len(lr_files):
        raise ValueError(f"GT文件夹({len(gt_files)}个文件)和LR文件夹({len(lr_files)}个文件)的文件数量不匹配")
    
    # 创建文件对列表
    file_pairs = list(zip(gt_files, lr_files))
    
    # 部分绑定，现在process_fn只需要gt_file和lr_file这两个参数
    process_fn = partial(process_worker, sizes=sizes,
                        resample=resample, lmdb_save=lmdb_save)

    if not lmdb_save:
        os.makedirs(out_path, exist_ok=True)
        os.makedirs('{}/lr_{}'.format(out_path, sizes[0]), exist_ok=True)
        os.makedirs('{}/hr_{}'.format(out_path, sizes[1]), exist_ok=True)
        os.makedirs('{}/sr_{}_{}'.format(out_path,
                    sizes[0], sizes[1]), exist_ok=True)
    else:
        env = lmdb.open(out_path, map_size=1024 ** 4, readahead=False)

    if n_worker > 1:
        # prepare data subsets
        multi_env = None
        if lmdb_save:
            multi_env = env

        file_subsets = np.array_split(file_pairs, n_worker)
        worker_threads = []
        wctx = WorkingContext(process_fn, lmdb_save, out_path, multi_env, sizes)

        # start worker processes, monitor results
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx, file_subsets[i]))
            proc.start()
            worker_threads.append(proc)
        
        total_count = str(len(file_pairs))
        while not all_threads_inactive(worker_threads):
            print("\r{}/{} images processed".format(wctx.value(), total_count), end=" ")
            time.sleep(0.1)

    else:
        total = 0
        for gt_file, lr_file in tqdm(file_pairs):
            imgs = process_fn(gt_file, lr_file) # imgs包含了同一个图片在不同分辨率下的结果
            lr_img, hr_img, sr_img = imgs
            total += 1
            index_str = str(total).zfill(6) # 生成图片的序号，6位数字
            
            if not lmdb_save:
                lr_img.save(
                    '{}/lr_{}/{}.png'.format(out_path, sizes[0], index_str))
                hr_img.save(
                    '{}/hr_{}/{}.png'.format(out_path, sizes[1], index_str))
                sr_img.save(
                    '{}/sr_{}_{}/{}.png'.format(out_path, sizes[0], sizes[1], index_str))
            else:
                with env.begin(write=True) as txn:
                    txn.put('lr_{}_{}'.format(
                        sizes[0], index_str).encode('utf-8'), lr_img)
                    txn.put('hr_{}_{}'.format(
                        sizes[1], index_str).encode('utf-8'), hr_img)
                    txn.put('sr_{}_{}_{}'.format(
                        sizes[0], sizes[1], index_str).encode('utf-8'), sr_img)
            if lmdb_save:
                with env.begin(write=True) as txn:
                    txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True,
                        help='GT图片文件夹路径')
    parser.add_argument('--lr_path', type=str, required=True,
                        help='LR图片文件夹路径')
    parser.add_argument('--out', '-o', type=str,
                        default='./dataset/processed')

    parser.add_argument('--size', type=str, default='256,256')
    parser.add_argument('--n_worker', type=int, default=3)
    parser.add_argument('--resample', type=str, default='bicubic')
    # default save in png format
    parser.add_argument('--lmdb', '-l', action='store_true')

    args = parser.parse_args()

    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]

    args.out = '{}_{}_{}'.format(args.out, sizes[0], sizes[1])
    prepare(args.gt_path, args.lr_path, args.out, args.n_worker,
            sizes=sizes, resample=resample, lmdb_save=args.lmdb)