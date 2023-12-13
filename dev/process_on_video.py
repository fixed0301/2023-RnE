import cv2
import torch
import numpy as np
import mediapipe as mp
from torch.utils.data import DataLoader
from dev.classes import MyDataset, Model

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def mouse_callback(event, x, y, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        process_clicked_area(param, clicked_point)

def process_clicked_area(img, clicked_point):
    global pose
    img = cv2.resize(img, (640, 640))

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
                x, y = int(x_and_y.x * 640), int(x_and_y.y * 640)
                idx += 1

            xy_list_list.append(xy_list)

            length = 45
            if len(xy_list_list) == length:
                dataset = []
                dataset.append({'key': 0, 'value': xy_list_list})
                dataset = MyDataset(dataset)
                dataset = DataLoader(dataset)
                xy_list_list = []
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



