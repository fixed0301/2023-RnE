import cv2
import torch
import numpy as np
import mediapipe as mp
import slack_sdk
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

status = 'None'
status_2 = 'None'

SLACK_TOKEN = 'xoxb-2329005458561-6373017132624-jlv24W6fJDFspZbGFM9XUTD2'
SLACK_CHANNEL = '19rne'


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
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.8)
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

    def __len__(self):  # 데이터셋의 길이 반환
        return len(self.x)

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

at_first = 1
def mouse_callback(event, x, y, flags, img, param=True):
    global clicked_point, selected_area_name
    global pose
    global at_first

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        if param:
            if at_first :
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
        #선택한 영역 이름
        at_first=0
        clicked_positions = []
        text_positions = [(50, 100), (150, 100), (250, 100)]
        for i, (text_x, text_y) in enumerate(text_positions, start=1):
            if i == 1:
                cv2.putText(img, 'slide', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif i == 2:
                cv2.putText(img, 'swing', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif i == 3:
                cv2.putText(img, 'else', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            clicked_positions.append((text_x, text_y))
        for i, (text_x, text_y) in enumerate(clicked_positions, start=1):
            if text_x < x < text_x + 100 and text_y - 30 < y < text_y + 10:
                if i==1 : selected_area_name='slide'
                elif i==2 : selected_area_name='swing'
                elif i==3 : selected_area_name = 'else'
                print("Mouse clicked at (x={}, y={}) - {}".format(x, y, selected_area_name))
                break
        else:
            print("Mouse clicked at (x={}, y={}) - Outside text area".format(x, y))


#초기 입력값
clicked_point = None
selected_area_name = ""
cnt=0
saved_areas = {}

#model_path = r"C:\Users\USER\Desktop\19rne\2023-RnE-main\save_by_loss\model_noswing_loss.pth"
video_path = r"C:\Users\USER\Desktop\video-2\20230905_162820.mp4"

model = Model()
#model.load_state_dict(torch.load(model_path))

cap = cv2.VideoCapture(video_path)

ret, first_img = cap.read()
cv2.namedWindow('First Image')

capp = cv2.VideoWriter('test vid.mp4', cv2.VideoWriter_fourcc(*'DIVX'), cap.get(cv2.CAP_PROP_FPS), (400, 700))


while True:

    first_img = cv2.resize(first_img, (400, 700))
    cv2.imshow('First Image', first_img)
    cv2.setMouseCallback('First Image', mouse_callback, first_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'): #창 닫기
        break
cv2.destroyAllWindows()



xy_list_list = []

while True:
    ret, image = cap.read()
    capp.write(image)
    if not ret:
        break
    image = cv2.resize(image, (400, 700))

    cv2.putText(image, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # mediapipe 실행하며 포즈 분류
    results = pose.process(image)

    if results.pose_landmarks:
        image.flags.writeable = True
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        xy_list = []
        for x_and_y in results.pose_landmarks.landmark:
            xy_list.append(x_and_y.x)
            xy_list.append(x_and_y.y)

        xy_list_list.append(xy_list)
        # print(a)
        # a+=1

        length = 10
        if len(xy_list_list) == length:
            #print('a dataset')
            # 데이터셋 구성
            dataset = [{'key': 0, 'value': xy_list_list}]
            dataset = MyDataset(dataset)
            dataset = DataLoader(dataset)

            for data, label in dataset:
                with torch.no_grad():
                    result = model(data)  # 동작 분류 결과
                    _, out = torch.max(result, 1)

                    if out.item() == 0:
                        status = 'backward'
                        if selected_area_name == 'slide' and status == 'backward':
                            if status != status_2:
                                Msg_bot(status)
                                status_2 = 'backward'
                    elif out.item() == 1:
                        status = 'swing'

                        if status != status_2:
                            Msg_bot(status)
                            status_2 = 'swing'
                    elif out.item() == 2:
                        status = 'slide'
                        if selected_area_name == 'slide' and status == 'slide':
                            if status != status_2:
                                Msg_bot(status)
                                status_2 = 'slide'
                    elif out.item() == 3:
                        status = 'swing'
                        if selected_area_name == 'swing' and status == 'swing' :
                            if status != status_2:
                                Msg_bot(status)
                                status_2 = 'swing'
                    elif out.item() == 4:
                        status = 'walk'
                        if status != status_2:
                            #Msg_bot(status)
                            status_2 = 'walk'

                    print(status)

            # 이제 처리한 데이터는 삭제
            xy_list_list = []

    else:
        global cnt
        cnt += 1
        #print('No person detected', cnt)

    cv2.imshow('Image', image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):  # 창 닫기
        break

cv2.destroyAllWindows()


# 저장된 영역과 좌표 출력
for area_name, area_info in saved_areas.items():
    print(f"영역명: {area_name}, 좌표: {area_info['coordinates']}")


# #%%
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
# print(length)

#%%
