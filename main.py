
import cv2
import torch
import numpy as np
import mediapipe as mp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn

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

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def mouse_callback(event, x, y, param):
    global clicked_point
    print('asdf')
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        process_clicked_area(param, clicked_point)

def process_clicked_area(img, clicked_point):
    global pose
    #img = cv2.resize(img, (640, 640))

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
        clicked_area = img[y:y + h, x:x + w]

        # mediapipe 실행하며 포즈 분류
        xy_list_list = []
        results = pose.process(cv2.cvtColor(clicked_area, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            xy_list = []
            idx = 0
            for x_and_y in results.pose_landmarks.landmark:
                xy_list.append(x_and_y.x)
                xy_list.append(x_and_y.y)
                #x, y = int(x_and_y.x * 640), int(x_and_y.y * 640)
                idx += 1

            xy_list_list.append(xy_list)

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
                        elif out.item() == 1:
                            status = 'sit'
                        elif out.item() == 2:
                            status = 'slide'
                        elif out.item() == 3:
                            status = 'swing'
                        elif out.item() == 4:
                            status = 'walk'

                        print(out.item())
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

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model





model_path = r'C:\Users\USER\Desktop\19rne\2023-RnE-main\saved_models_2model_weight.pth'
model = load_model(Model(), model_path)


model.eval()
status = 'None'
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(r"C:\Users\USER\Desktop\19rne\video_data\swing_5.mp4")  # 0은 기본 카메라, 필요에 따라 변경 가능

while True:
    ret, img = cap.read()

    if not ret:
        break

    img = cv2.resize(img, (200, 400))
    cv2.imshow('Real-time Analysis', img)

    cv2.setMouseCallback('Real-time Analysis', mouse_callback, param=img)


    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()


#%%
