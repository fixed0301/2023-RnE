import cv2
import torch
import numpy as np
import mediapipe as mp
import slack_sdk
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#from dev.alarm import Msg_bot

status = 'None'
status_2 = 'None'

SLACK_TOKEN = 'xoxb-2329005458561-2340049133872-QwqGVz6Io1ZQVaQEf6h0naph'
SLACK_CHANNEL = '정렌이'


def Msg_bot(status):
    slack_token = SLACK_TOKEN   #slack bot token
    channel = SLACK_CHANNEL
    message = chat[status]
    client = slack_sdk.WebClient(token=slack_token)
    client.chat_postMessage(channel=channel, text=message)

chat = {'backward' : '미끄럼틀을 역행하고 있습니다.',
        'sit' : '앉아있습니다.',
        'slide' : '미끄럼틀을 내려오고 있습니다.',
        'swing' : '그네를 타고 있습니다.',
        'walk' : '걷고 있습니다.',
        'collision' : '충돌 위험이 있습니다'} # 1p : swing / slide, 2p : any



mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class MyDataset(Dataset):
    def __init__(self, dataset): #모든 행동을 통합한 df가 들어가야함
        self.x = []
        self.y = []
        for dic in dataset:
            self.y.append(dic['key']) #key 값에는 actions 들어감
            self.x.append(dic['value']) #action마다의 data 들어감

    def __getitem__(self, index): #index는 행동의 index
        data = self.x[index] # x에는 꺼내 쓸 (행동마다 45개 묶음프레임)의 데이터
        label = self.y[index]
        return torch.Tensor(np.array(data)), torch.tensor(np.array(int(label)))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm1 = nn.LSTM(input_size=99, hidden_size=128, num_layers=1, batch_first=True) #input은  45 * 3(x, y z)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0, 1)
        self.lstm4 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm5 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm6 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0, 1)
        self.lstm7 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 5) #분류할 클래스 5가지

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout1(x)
        x, _ = self.lstm4(x)
        x, _ = self.lstm5(x)
        x, _ = self.lstm6(x)
        x = self.dropout2(x)
        x, _ = self.lstm7(x)
        x = self.fc(x[:, -1, :])
        return x



def mouse_callback(event, x, y, flags, img, param=True):
    global clicked_point, selected_area_name
    global pose

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        if param:
            selected_area_name = input("영역 이름 입력: ")



        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        clicked_hsv_color = hsv_image[clicked_point[1], clicked_point[0]]

        # 클릭한 픽셀과 유사한 색상 추출
        color_range = 15  # 색상 범위
        lower_bound = np.array([clicked_hsv_color[0] - color_range, 50, 50])
        upper_bound = np.array([clicked_hsv_color[0] + color_range, 255, 255])

        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        result = cv2.bitwise_and(img, img, mask=mask)

        # 연속된 픽셀을 묶기
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 클릭한 픽셀이 포함된 무리 찾기
        clicked_cluster = None
        for contour in contours:
            if cv2.pointPolygonTest(contour, clicked_point, False) >= 0:
                clicked_cluster = contour
                break

        if clicked_cluster is not None:

            x, y, w, h = cv2.boundingRect(clicked_cluster) # 자르기
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            clicked_area = img[y:y + h, x:x + w]
            saved_areas[selected_area_name] = {'coordinates': (x, y, w, h)}



#초기 입력값
clicked_point = None
selected_area_name = ""
cnt=0
saved_areas = {}

cap = cv2.VideoCapture(r"C:\Users\USER\Desktop\19rne\video_data\slide_14.mp4")

ret, first_img = cap.read()
cv2.namedWindow('First Image')

while True:

    first_img = cv2.resize(first_img, (400, 700))
    cv2.imshow('First Image', first_img)
    cv2.setMouseCallback('First Image', mouse_callback, first_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'): #창 닫기
        break
cv2.destroyAllWindows()

while True:
    ret, image = cap.read()

    if not ret:
        break
    image = cv2.resize(image, (400, 700))

    cv2.putText(image, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



    # mediapipe 실행하며 포즈 분류
    xy_list_list = []
    results = pose.process(image)


    if results.pose_landmarks:
        print('processing')
        image.flags.writeable = True
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


        xy_list = []
        idx = 0
        for x_and_y in results.pose_landmarks.landmark:
            xy_list.append(x_and_y.x)
            xy_list.append(x_and_y.y)
            #x, y = int(x_and_y.x * 640), int(x_and_y.y * 640)
            idx += 1

        xy_list_list.append(xy_list)
        print(xy_list_list)


        length = 45
        if len(xy_list_list) == length:
            dataset = []
            dataset.append({'key': 0, 'value': xy_list_list})
            dataset = MyDataset(dataset)
            dataset = DataLoader(dataset)

            for data, label in dataset:
                with torch.no_grad():
                    result = Model(data) # 동작 분류 결과
                    _, out = torch.max(result, 1)

                    if out.item() == 0:
                        status = 'backward'
                        if status!=status_2 : Msg_bot(status)
                        status_2 = 'backward'
                    elif out.item() == 1:
                        status = 'sit'
                        if status!=status_2 : Msg_bot(status)
                        status_2 = 'sit'
                    elif out.item() == 2:
                        status = 'slide'
                        if status!=status_2 : Msg_bot(status)
                        status_2 = 'slide'
                    elif out.item() == 3:
                        status = 'swing'
                        if status!=status_2 : Msg_bot(status)
                        status_2 = 'swing'
                    elif out.item() == 4:
                        status = 'walk'
                        if status!=status_2 : Msg_bot(status)
                        status_2 = 'walk'

                    print(out.item())
    else:
        global cnt
        cnt+=1
        print('No person detected', cnt)

    cv2.imshow('Image', image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'): #창 닫기
        break
cv2.destroyAllWindows()

# 저장된 영역과 좌표 출력
for area_name, area_info in saved_areas.items():
    print(f"영역명: {area_name}, 좌표: {area_info['coordinates']}")




#%%
