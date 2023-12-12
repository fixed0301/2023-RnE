import cv2
import numpy as np

# 마우스 클릭
def mouse_callback(event, x, y, flags, param):
    global clicked_point, selected_area_name
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        selected_area_name = input("영역 이름 입력: ")
        process_clicked_area()

# 클릭한 좌표와 유사한 색상을 추출 & 연속된 픽셀을 묶는 함수
def process_clicked_area():
    if clicked_point is not None:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        clicked_hsv_color = hsv_image[clicked_point[1], clicked_point[0]]

        color_range = 15  # 색상 범위 
        lower_bound = np.array([clicked_hsv_color[0] - color_range, 50, 50])
        upper_bound = np.array([clicked_hsv_color[0] + color_range, 255, 255])

        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        result = cv2.bitwise_and(image, image, mask=mask)

        # 연속된 픽셀을 하나로 묶음
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        clicked_cluster = None # 클릭한 픽셀이 포함된 무리 찾기
        for contour in contours:
            if cv2.pointPolygonTest(contour, clicked_point, False) >= 0:
                clicked_cluster = contour
                break

        if clicked_cluster is not None: # 영역 잘라내기
            x, y, w, h = cv2.boundingRect(clicked_cluster)
            clicked_area = image[y:y + h, x:x + w]

            saved_areas[selected_area_name] = {'coordinates': (x, y, w, h)} # 추출한 영역을 임의로 정한 제목으로 저장 및 좌표 저장

            cv2.imshow('Clicked Area - ' + selected_area_name, clicked_area)

#초기 입력값
clicked_point = None
selected_area_name = ""
image = cv2.imread('test1.jpg') 
saved_areas = {}

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    cv2.imshow('Image', image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'): #창 닫기
        break
cv2.destroyAllWindows()

# 저장된 영역과 좌표 출력
for area_name, area_info in saved_areas.items():
    print(f"영역명: {area_name}, 좌표: {area_info['coordinates']}")