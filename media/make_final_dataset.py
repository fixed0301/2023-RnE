import os
import pandas as pd
actions_csv_dir = '../csv_1031/'
dataset = []

label_mapping = {'backward': 0,
                 'sit': 1,
                 'slide': 2,
                 'swing': 3,
                 'walk' : 4
                 }
length = 45
def map_action_to_label(csv_name):
    for action, label in label_mapping.items():
        if action in csv_name.split('_')[0]:
            return label
    return -1

for action_csv in os.listdir(actions_csv_dir):
    action_df = pd.read_csv(os.path.join(actions_csv_dir, action_csv))
    label = map_action_to_label(action_csv)
    if label != -1:
        for idx in range(0, len(action_df), int(length / 2)):
            seq_df = action_df[idx: idx + length] #길이만큼 데이터 자른 것(즉 length 만큼의 프레임)
            if len(seq_df) == length: # 딱 length에 개수 맞춰서 끊어서 넣으려고
                dataset.append({'key': label, 'value': seq_df}) # key에 slide, value에는 묶음 프레임 만큼이 담기겠네
    #최종적으로 dataset에는 행동별로 dictionary 가 만들어져 들어간다.
print('asdf')
print(len(dataset))
