import argparse
import os
import glob
import numpy as np
import re
import json
import matplotlib
import setproctitle
import functools
import multiprocessing as mp
import matplotlib.pyplot as plt
import sys

sys.path.append(os.getcwd())

from tqdm import tqdm
from IPython import embed #强大的ipython交互式shell

from multiprocessing import Pool

mp.set_start_method('spawn',True)

def embeded_numbers(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)
    return int(pieces[1])


def embeded_numbers_results(s):
    re_digits = re.compile(r'(\d+)')#r的意思是不转义，即\表示原样的\,否则有可能被视图按 \d 为一个字符解析转移
    pieces = re_digits.split(s) # '\d'是匹配数字字符[0-9], + 表示匹配一个或多个，比如 ‘1’， ‘34’ ， ‘2434’
    return int(pieces[-2])


def cal_iou(box1, box2):
    """
    :param box1: x1,y1,w,h
    :param box2: x1,y1,w,h
    :return: iou
    """
    x11 = box1[0]
    y11 = box1[1]
    x21 = box1[0] + box1[2] - 1
    y21 = box1[1] + box1[3] - 1
    area_1 = (x21 - x11 + 1) * (y21 - y11 + 1)

    x12 = box2[0]
    y12 = box2[1]
    x22 = box2[0] + box2[2] - 1
    y22 = box2[1] + box2[3] - 1
    area_2 = (x22 - x12 + 1) * (y22 - y12 + 1)

    x_left = max(x11, x12)
    x_right = min(x21, x22)
    y_top = max(y11, y12)
    y_down = min(y21, y22)

    inter_area = max(x_right - x_left + 1, 0) * max(y_down - y_top + 1, 0)
    iou = inter_area / (area_1 + area_2 - inter_area)
    return iou

def cal_success(iou):
    success_all = []
    overlap_thresholds = np.arange(0, 1.05, 0.05)
    for overlap_threshold in overlap_thresholds:
        success = sum(np.array(iou) > overlap_threshold) / len(iou)
        success_all.append(success)
    return np.array(success_all)

def cal_precision(gt_center, result_center):

    thresholds_error = np.arange(0, 51, 1)
    n_frame = len(gt_center)
    success = np.zeros(len(thresholds_error))
    dist = np.sqrt(np.sum(np.power(gt_center - result_center, 2), axis=1))
    for i in range(len(thresholds_error)):
        success[i] = sum(dist <= thresholds_error[i]) / float(n_frame)
    return success

def cal_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

#多线程
# def worker(video_paths, model_path):
#     results_ = {}
#     for video_path in tqdm(video_paths, total=len(video_paths)):
#         groundtruth_path = video_path + '/groundtruth_rect.txt'
#         assert os.path.isfile(groundtruth_path), 'groundtruth of ' + video_path + ' doesn\'t exist'
#         with open(groundtruth_path, 'r') as f:
#             boxes = f.readlines()
#         if ',' in boxes[0]:
#             boxes = [list(map(int, box.split(','))) for box in boxes]
#         else:
#             boxes = [list(map(int, box.split())) for box in boxes]
#         result = run_SiamFC(video_path, model_path, boxes[0])
#         results_box_video = result['res']
#         results_[os.path.abspath(model_path)][video_path.split('/')[-1]] = results_box_video
#         return results_

if __name__ == '__main__':

    program_name = os.getcwd().split('/')[-1] #得到当前
    setproctitle.setproctitle('test ' + program_name) #获取和修改进程的名字

    parser = argparse.ArgumentParser(description='Test some models on OTB2015 or OTB2013')
    parser.add_argument('--model_paths', '-ms', dest='model_paths', nargs='+',
                        help='the path of models or the path of a model or folder',default='./models/siamrpn_38.pth')
    parser.add_argument('--videos', '-v', dest='videos',default='100')  # choices=['tb50', 'tb100', 'cvpr2013']
    parser.add_argument('--save_name', '-n', dest='save_name', default='./json/result.json') #保存的路径
    parser.add_argument('--data_path', '-d', dest='data_path', default='./data/OTB100/') #数据集的路径
    args = parser.parse_args()

    model_path =args.model_paths 

    # ------------ prepare data  -----------
    data_path = args.data_path
    if '50' in args.videos:
        direct_file = data_path + 'tb_50.txt'
    elif '100' in args.videos:
        direct_file = data_path + 'tb_100.txt' #读取视频文件名和对应的属性列表
    elif '13' in args.videos:
        direct_file = data_path + 'cvpr13.txt'
    else:
        raise ValueError('videos setting wrong')
    with open(direct_file, 'r') as f:
        direct_lines = f.readlines() #按行读取
    video_names = np.sort([x.split('\t')[0] for x in direct_lines]) #每一行按照tab符来分割，获取所有视频名字
    video_paths = [data_path + x for x in video_names] #获取相对路径
    #print(video_names)#打印输出视频名字
    
    # ------------ starting validation  -----------
    result_file=os.path.join(args.save_name)
    results = {}    

    if not os.path.exists(result_file):
        # 单线程处理视频数据集    
        for video_path in tqdm(video_paths, total=len(video_paths)):
           # if video_path=='./data/OTB100/Tiger1':
            groundtruth_path = video_path + '/groundtruth_rect.txt'
            assert os.path.isfile(groundtruth_path), 'groundtruth of ' + video_path + ' doesn\'t exist'
            with open(groundtruth_path, 'r') as f:
                boxes = f.readlines()
            if ',' in boxes[0]:
                boxes = [list(map(int, box.split(','))) for box in boxes] #将ground数据集转换成int类型，然后转换成list列表
            else:
                boxes = [list(map(int, box.split())) for box in boxes]
            boxes = [np.array(box) - [1, 1, 0, 0] for box in boxes] #将matlab中数值从1开始，转换成python中从0开始
            #result = run_SiamRPN(video_path, model_path, boxes[0])
            result = run_SiamRPN(video_path, model_path, boxes[0])
            result_boxes = [np.array(box) + [1, 1, 0, 0] for box in result['res']] #再转换成matlab格式
            #results[os.path.abspath(model_path)][video_path.split('/')[-1]] = [box.tolist() for box in result_boxes]
            results[video_path.split('/')[-1]] = [box.tolist() for box in result_boxes]#转成list
        json.dump(results, open(args.save_name, 'w')) #把数据类型转换成字符串，并存储在文件中
           # break
    #多线程处理视频数据集
    # with Pool(processes=mp.cpu_count()) as pool:
    #     for ret in tqdm(pool.imap_unordered(
    #             functools.partial(worker, video_paths), model_paths), total=len(model_paths)):
    #         results.update(ret)
    # json dumps 把数据类型转换成字符串； dump 把数据类型转成字符串并存储在文件中；loads 把字符串转换成数据类型； load把文件打开从字符串转换成数据类型
    # json可以在不同语言之间交换数据，而pickle只在python之间使用，json只能把常用的数据类型序列化 （列表，字典，字符串，数字）；日期，对象类 json不适合
    # pickle可以序列化所有的数据类型
    else:

        results=json.load(open(args.save_name,'r'))

    # ------------ starting evaluation  -----------
    results_auc = {}
    results_pre={}
    success_all_video = []
    precision_all_video = []
    #for model in sorted(list(results.keys()), key=embeded_numbers_results):
    for video in sorted(list(results.keys())):
        #if video=='Diving':
        result_boxes = results[video]#得到测试的box
        # 获取groundtruth
        with open(data_path + video + '/groundtruth_rect.txt', 'r') as f:
            result_boxes_gt = f.readlines()
        if ',' in result_boxes_gt[0]:
            result_boxes_gt = [list(map(int, box.split(','))) for box in result_boxes_gt]#读取txt文件的操作
        else:
            result_boxes_gt = [list(map(int, box.split())) for box in result_boxes_gt]
        result_boxes_gt = [np.array(box) for box in result_boxes_gt]#得到groundtruth box
        
        #计算Overlap

        iou = list(map(cal_iou, result_boxes, result_boxes_gt)) #计算每一帧的iou
        success = cal_success(iou) #
        success_all_video.append(success)
        auc = np.mean(success)  #   
        results_auc[video]= auc

        #计算Precision
        boxes=np.array(result_boxes).astype(np.float)
        boxes_gt=np.array(result_boxes_gt).astype(np.float)
        result_center =cal_center(boxes) #得到 center_point
        gt_center = cal_center(boxes_gt) #得到 center_point
        precision=cal_precision(gt_center,result_center)
        pre=precision[21]
        precision_all_video.append(pre)
        results_pre[video] = pre

    
    mean_auc='%.04f' % np.mean(success_all_video)
    mean_pre='%.04f' % np.mean(precision_all_video)

    results_auc['all_auc'] = mean_auc
    results_pre['all_pre'] = mean_pre

    print( 'mean_auc:', mean_auc)
    print( 'mean_pre:', mean_pre)

    json.dump(results_auc, open('./json/eval_auc_result.json', 'w'))
    json.dump(results_auc, open('./json/eval_pre_result.json', 'w'))

    
    