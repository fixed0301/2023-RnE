import cv2
import media as mp
import cv2
import torch
from dev.alarm import Msg_bot
from dev.process_on_video import mouse_callback

model_filename = "best_model_fold_4.pth"
model_path = '../saved_models/' + model_filename
model = load_model(Model(), model_path)

model.eval()
status = 'None'
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)  # 0은 기본 카메라, 필요에 따라 변경 가능

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback, param=cap)

while True:
    ret, img = cap.read()
    if not ret:
        break

    cv2.putText(img, status, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == 27:  # ESC 키로 종료
        break

cv2.destroyAllWindows()





        Msg_bot(status) # 알림 전송

    cv2.putText(img, status, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
    cv2.imshow('Real-time Analysis', img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
