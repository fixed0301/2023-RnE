import re

import cv2
import glob
import json
import csv
import numpy as np
import os

#json ex는 body_25니까 coco로 바꿔서 실행

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
actions = ['backward', 'lie', 'sit', 'slide', 'stand', 'swing', 'walk']
def load_images(path): #이미지 저장
    path = os.path.join(path, '*')
    files = glob.glob(path)
    files.sort()

    X_raw = []
    for file in files:
        image = cv2.imread(file)
        X_raw.append(np.array(image))

    return X_raw


def from_json(path): #path로 json 저장해두면 불러와서 사용
    file = open(path, 'r', encoding='utf-8')
    return json.load(file)


def extract_openpose_anns(ann_json):
    def extract_keypoints(ann_json):
        X = []
        Y = []
        C = []
        id = 0
        while id < len(ann_json):
            if ann_json[id] != 0:
                X.append(ann_json[id]) #여기에 전처리 과정을 추가
                Y.append(ann_json[id + 1])
                C.append(ann_json[id + 2]) #confidence score
                id += 3
            else:
                X.append(None)
                Y.append(None)
                C.append(None)
                id += 3

        return np.array([X, Y, C])

    kp_pose = extract_keypoints(ann_json['people'][0]['pose_keypoints_2d']) #array

    pose = {}
    i = 0
    for key in BODY_PARTS.keys():
        name_x = f'{key}' + '_x'
        name_y = f'{key}' + '_y'

        pose[f'{name_x}'] = kp_pose[0][i]
        pose[f'{name_y}'] = kp_pose[1][i]
        i += 1
    return pose


def extract_pose_annotations(path): #path 안 json 파일꺼내서 필요한 pose 좌표 dict으로 정리
    path = os.path.join(path, '*')
    files = glob.glob(path)
    # re.sub(r'[^0-9]', '', files)
    # files.sort(key=lambda x: x.lstrip('0'))

    Y_raw = []
    for file in files:
        ann_json = from_json(file)
        ann = extract_openpose_anns(ann_json)
        #ann은 dict
        # 정규화한 좌표 추가
        Y_raw.append(ann)

    return Y_raw

#test
with open('keypoints.json') as json_file:
    data = json.load(json_file)
pose = extract_openpose_anns(data) # {'Nose_x': 951.139, 'Nose_y': 480.116,...

#convert to csv

# for action in actions:
#     Y_raw = extract_pose_annotations(f'../landmark-json/{action}') #json 폴더마다 꺼내기
#     for landmarkSet in Y_raw: #landmarkSet written in dictionary
#         with open(f'../landmark-csv/landmark-{action}.csv', 'w') as f: #하나의 csv 파일에 몇개나 저장할까.. backward에 대한 csv,
#             w = csv.writer(f)
#             w.writerow(landmarkSet.keys())
#             w.writerow(landmarkSet.values())
#             f.close()


