import cv2
import numpy as np

def mouse_callback(event, x, y):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

def process_clicked_area():
    if clicked_point is not None:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        clicked_hsv_color = hsv_image[clicked_point[1], clicked_point[0]]

        # 클릭한 픽셀과 유사한 색상 추출
        color_range = 15  # 색상 범위
        lower_bound = np.array([clicked_hsv_color[0] - color_range, 50, 50])
        upper_bound = np.array([clicked_hsv_color[0] + color_range, 255, 255])

        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        result = cv2.bitwise_and(image, image, mask=mask)

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
            clicked_area = image[y:y + h, x:x + w]
            # cv2.imshow('Clicked Area', clicked_area)
            return (x, y, w, h)
        else:
            return 'No cluster found'


clicked_point = None
image = cv2.imread('test1.jpg')

cv2.setMouseCallback('Image', mouse_callback)

while True:
    cv2.imshow('Image', image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        break

cv2.destroyAllWindows()