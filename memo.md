구분	역할
🧩 Audio-Visual Synchronization Model	오디오와 비디오가 시간적으로 맞는지 예측
🧱 Segment-level Feature Extractor	영상과 오디오에서 특징(feature)을 추출
🧠 Synchronizability Prediction Model	“이 영상이 얼마나 동기화 가능한지(동기화 난이도)” 평가


이어서 할 것
 python example.py --exp_name "24-01-04T16-39-21" --vid_path "/data/10s.mp4" --offset_sec 1.6
 잘 작동하는지 확인 // 완료

 이 후 



 🧠 Synchronizability Prediction Model  이란?

 영상이 동기화가 얼마나 가능한지 가능성에 대한평가 

 드럼 연주 같은경우 쉬움,  배경음 있는 풍경은 어려움    같은 방식

 >>> 시각-청각 일치 확률이 낮은 샘플  필터링 정도에 써먹을 수 있음
 >>> greathi 데이터셋 한정으론 hit 프레임마다 높은 값이 나올 것이고 이를 이용해 onset 검출 feature로 사용할 수도 있기는 함.



 계획1
 visdataset에 대해 synchformer(audioset-pretrained)을 이용해 각 프레임 별 synchronizability 점수를 계산
 이를 시각화 + 학습용 데이터로 활용

 하지만 다른 데이터셋, nc의 게임 영상에 대해서도 잘 작동할지는 모름.