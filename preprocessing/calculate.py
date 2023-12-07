import csv
import glob
import pandas as pd
import glob
import os

input_path = r'../landmark-csv' #모든 프레임마다 추출한 csv 파일
output_file = r'../landmark-csv-processed' #5프레임씩 묶어 연산한 것(lstm input)
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

def mean_col(name):
    xs = df[f'{name}'][cut_start:cut_end]
    total_sum = xs.sum()
    mean_xs = round(total_sum / xs.count(), 3)
    return mean_xs

def neck_knee(name):
    xs = df[f'{name}'][cut_start:cut_end]


def process_csv(path):
    global df, cut_start, cut_end
    file_list = glob.glob(os.path.join(input_path, '.csv'))
    col_name = list(BODY_PARTS.keys())
    all_dataFrame = [] # 데이터 프레임을 저장할 리스트
    for file in file_list:
        file_name = os.path.basename(file)
        df = pd.read_csv(file)
        cut_start = 0
        cut_end = 5
        for name in col_name:
            xs = df[f'{name}'][cut_start:cut_end]
            mean_xs = mean_col(name)
            mean_h = mean_col()
            value = (xs - mean_xs)/h

            cut_start += 5
            cut_end += 5
            h = #neck-knee 평균 길이
            value = xs - mean_xs / h
            data = {'filename': [file_name], 'average': [mean]}
            all_dataFrame.append(pd.DataFrame(data=data))


process_csv(input_path)
#    All_data = pd.concat(all_dataFrame, axis=0, ignore_index=True) # 데이터 프레임들을 병합한다.
#    All_data.to_csv(output_file, index=False)

