import cv2
#좌표가 이상한데 찍힌다. 코드 복붙인데 import 해서 사용하는 조건에서만 오류나는게 이상하긴 함
#그래도 별 상관 없음 좌표는 json으로 저장할거니까

def outputkeypoints(frame, proto_file, weights_file):
    global points
    points = []

    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # 입력 이미지의 사이즈 정의
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (frame_width, frame_height), (0, 0, 0), swapRB=False,
                                       crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    out = net.forward()

    out = out[:, :19, :, :] #네트워크의 정해진 ouput이 있어서 마음대로 벡터 연결시키면 안됨. 우선 18 keypoints 얻고 필요시 제거
    out_height = out.shape[2]
    # The fourth dimension is the width of the output map.
    out_width = out.shape[3]
    thres = 0.5
    for i in range(len(BODY_PARTS)):
        prob_map = out[0, i, :, :]
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정
        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)

        if prob > thres:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        else:
            points.append(None)
    return frame

def get_joints(frame):
    circled_frame = outputkeypoints(frame, proto_file, weights_file)
    return points

#COCO model
# 440000 caffe
BODY_PARTS = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}


# POSE_PAIRS = [['Nose', 'Neck'], ['Neck', 'LShoulder'], ['Neck', 'RShoulder'], ['LShoulder', 'LElbow'], ['RShoulder', 'RElbow'], ['LElbow', 'LWrist'], ['RElbow', 'RWrist'], ['Neck', 'LHip'], ['Neck', 'RHip'], ["LHip", 'LKnee'], ["RHip", "RKnee"], ['LKnee', 'LAnkle'], ['RKnee', 'RAnkle']] #use only 14pairs
POSE_PAIRS = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                   [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]

proto_file = 'E:\\2023\\2023_1_1\\rne\openpose\models\pose\coco\pose_deploy_linevec.prototxt'
weights_file = 'E:\\2023\\2023_1_1\\rne\openpose\models\pose\coco\pose_iter_440000.caffemodel'

im = cv2.imread("E:\im\\fullshot.jpg")
colored_frame = outputkeypoints(im, proto_file, weights_file)
print(points)
# for i in points:
#     if i:
#         cv2.circle(im, i, 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
#         cv2.putText(im, str(i), i, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

cv2.imshow('sf', colored_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()