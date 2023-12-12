# TODO
- [ ] 영상 촬영 + json 저장
- [x] 영상에서 프레임 추출, 크기 정규화. 4초 영상 기준 5frame 씩 묶으면 len(sequencedata) = 30 정도 
  ex) videodata/(backward, slide, swing, sit, walk, standing, lie)/backward001.mp4
- [x] json to csv. 
- [ ] 7 가지 pose 에 대해 20번 정도 sequence 별로 skeleton data 저장된 csv 파일 생성
- [ ] Null 값 이전 프레임 값으로 채우기 (값이 너무 적을 때 프레임 제거 기준 필요)
- [ ] 정규화 계산대로 입력 좌표 정규화 
- [ ] csv로부터 읽어서 input 형식 맞춰 모델 구성
- [ ] 사람 좌표에 대해 놀이터 기구 영역 구분. 
      이진화 혹은 HSV 색공간 기반 분리, (분리 후 각 기구에 대한 identification 필요! 자동화하려면 무조건 yolo 사용해야하나 최소 1000 ~ 1500장 이미지 수집과 수동 라벨링하기. )  카메라가 고정되었다고 가정하면 가장 처음에만 사람이 영역 지정하도록.
<<<<<<< HEAD
      
=======

# New TODO!
- [ ] 화면에서 영역별로 사람이 닿으면 
>>>>>>> e27158fc5c4241c805d5e4b268f7e3388188ef4c

# 기존 계획 변경사항

- 모델이 추론하는 output 형식이 정해져 있어서 임의로 골격 연결관계 바꾸면 안됐음. 
  따라서 csv로 옮긴 후 불필요한 값 제거 필요
- 행동 학습시킬 때는 한 명에 대한 skeleton data 만 사용.

# 추가사항
- 미끄럼틀 위에 서있으면 '미끄럼틀을 서서 타면 위험합니다' 같은 경고를 스피커로 틀기

