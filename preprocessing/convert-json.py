import csv
import cv2
import glob
import json
import numpy as np
import os
import pandas as pd
from dev.create_csv import csv
from dev.calculate import preprocess_2


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip":8, "RHip": 9,
               "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
               "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
               "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe":23, "RHeel": 24}

POSE_PAIRS = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                      [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                      [11, 24], [22, 24], [23, 24]]

actions = ['backward', 'lie', 'sit', 'slide', 'stand', 'swing', 'walk']

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

def extract_pose_annotations(path): #folder path 안 json 파일꺼내서 필요한 pose 좌표 dict으로 정리
    path = os.path.join(path, '*')
    files = glob.glob(path)
    Y_raw = []
    for file in files:
        ann_json = from_json(file)
        ann = extract_openpose_anns(ann_json)
        # ann은 dictionary
        # 정규화한 좌표 추가
        Y_raw.append(ann)

    return Y_raw


df_list = []

json_action_folder = f'../landmark-json/backward'
for action_file in os.listdir(json_action_folder):
    file_path = os.path.join(json_action_folder, action_file)

    try:
        with open(file_path) as json_file:
            data = json.load(json_file)
            pose = extract_openpose_anns(data) #{'Nose_x': 951.139, 'Nose_y': 480.116,...
            temp_df = pd.DataFrame(pose, index=[0])
            df_list.append(temp_df)
    except IndexError:
        print('으아 인덱스가 이상하게 벗어났어요:', action_file)

df1 = pd.concat(df_list, ignore_index=True) #[418 rows x 36 columns]
'''
행마다 if (RKnee and LKnee is None) or (Neck_y is None) : delete row
그리고 탐지된 개수가 총 36가지 중 18개보다 작으면 제거
남은 결측값은 이전 프레임걸로, 처음 행은 뒤의 프레임걸로
'''
df1 = df1.dropna(subset=['RKnee_y', 'LKnee_y'], how='all')
df1 = df1.dropna(subset=['Neck_y']) # [418 rows x 36 columns]

df1.fillna(method='ffill', inplace=True)
df1.iloc[:, 0].fillna(method='bfill', inplace=True)
df1.dropna(inplace=True)

#print(df1.isnull().sum(axis = 0)) #null 개수 세보자

# print(final_preprocess)
#csv(df1, 'slide_test_0919')

# 주어진 데이터를 pandas DataFrame으로 불러오기
data = df1

# 화면 크기 설정
width, height = 900, 600  # 원하는 해상도로 설정

# 빈 화면 생성
frame = cv2.imread("E:\im\circle.png")

for idx in range(len(data)):
# 데이터프레임의 한 행 선택 (예: 첫 번째 행)
    row = data.iloc[idx]
    frame = cv2.imread("E:\im\circle.png")

    # 관절 좌표를 리스트로 추출
    x = row.values[::2]  # 홀수 인덱스는 X 좌표
    y = row.values[1::2]  # 짝수 인덱스는 Y 좌표
    # 점 색상 설정 (예: 랜덤 색상)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    # 관절을 다른 색상으로 표시
    for i in range(len(x)):
        color = colors[i % len(colors)]  # 색상 순환
        cv2.circle(frame, (int(y[i]), int(x[i])), 10, color, -1)
    for pair in POSE_PAIRS:
        part_a = pair[0]
        part_b = pair[1]
        cv2.line(frame, (int(y[part_a]), int(x[part_a])), (int(y[part_b]), int(x[part_b])), (0, 255, 0), 3)

    # 화면에 프레임 표시
    cv2.imshow('Skeleton Visualization', frame)
    key = cv2.waitKey(100)  # 아무 키나 누를 때까지 대기
    if key == 27:
        break
cv2.destroyAllWindows()





