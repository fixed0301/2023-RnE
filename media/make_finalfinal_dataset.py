import os
import cv2
import mediapipe as mp
import pandas as pd

data_folder = r'C:\Users\USER\Desktop\19rne\video_data'

def csv(df, output_name):
    df.to_csv(r'C:\Users\USER\Desktop\19rne\2023-RnE-main\csv_1213'+ output_name +'.csv', index=False)

def create_df(video_name):
    video_path = os.path.join(data_folder, video_name)
    # 33개 landmark
    BODY_PARTS = { "Nose": 0, "LEye_in": 1, "LEye": 2, "LEye_out": 3, "REye_in": 4, "REye": 5, "REye_out": 6, "LEar": 7, "REar":8,
                   "LMouth": 9, "RMouth": 10, "LShoulder": 11, "RShoulder": 12,
                   "LElbow": 13, "RElbow": 14, "LWrist": 15, "RWrist": 16,
                   "LPinky": 17, "RPinky": 18, "LIndex": 19, "RIndex": 20,
                   "LThumb": 21, "RThumb": 22, "LHip": 23, "RHip": 24,
                   "LKnee": 25, "RKnee": 26, "LAnkle": 27, "RAnkle": 28,
                   "LHeel": 29, "RHeel": 30, "LFIndex": 31, "RFIndex": 32}

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5)

    xy_name = []
    for key in BODY_PARTS.keys():
        name_x = f'{key}_x'
        name_y = f'{key}_y'
        name_z = f'{key}_z'
        xy_name.append(name_x)
        xy_name.append(name_y)
        xy_name.append(name_z)

    df = pd.DataFrame(columns=xy_name)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (400, 700))
        results = pose.process(image)
        if not results.pose_landmarks:
            continue

        # pose data 저장
        x = [] # 2*33차원 리스트

        # k = 좌표 번호
        # results.pose_landmarks.landmark[k].x/y/z/visibility

        for k in range(33):
            if k == 1 or 3 or 4 or 6 or 7 or 8 or 9 or 10:
                continue
            else:
                if results.pose_landmarks:
                    x.append(results.pose_landmarks.landmark[k].x)
                    x.append(results.pose_landmarks.landmark[k].y)
                    x.append(results.pose_landmarks.landmark[k].z)
                    # x.append(results.pose_landmarks.landmark[k].visibility)
                else:
                    x.append(None)
                    x.append(None)
                    x.append(None)

            new_row = pd.Series(x, index=df.columns)
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
            csv(df, video_name)
    return


csv(df, 'mediapipe_test_lm')
print('Saved csv')


# Draw the pose annotation on the image.
# image.flags.writeable = True
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# mp_drawing.draw_landmarks(
#     image,
#     results.pose_landmarks,
#     mp_pose.POSE_CONNECTIONS,
#     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#
# cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))