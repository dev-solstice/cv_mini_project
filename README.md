# Stickman2Picture
졸라맨 그림을 포즈(관절 좌표)로 변환한 뒤, ControlNet으로 프롬프트에 맞는 캐릭터 이미지를 생성하는 미니프로젝트

## 데이터셋
- COCO Keypoints: https://cocodataset.org/#keypoints-2020
- Quick, Draw! (yoga 카테고리): https://github.com/googlecreativelab/quickdraw-dataset

## 참조 코드
- ControlNet: https://github.com/lllyasviel/ControlNet-v1-1-nightly
- Diffusers: https://github.com/huggingface/diffusers
- controlnet_aux: https://github.com/huggingface/controlnet_aux
- Simple Baselines: https://github.com/microsoft/human-pose-estimation.pytorch

## 파일 설명
- reference/controlnet_openpose_demo.py : 기존 방식 (사람 사진 → OpenPose → 이미지 생성)
- reference/make_stickman_dataset.py : 추가 구현용 학습 데이터 생성 (COCO 관절 좌표 → 졸라맨 이미지 + 정답 좌표)

## 추가 구현 내용
기존 OpenPose 검출기는 실제 사람 사진만 인식하므로, 졸라맨 → 관절 좌표 변환 모델을 자체 데이터셋으로 학습하여 ControlNet 앞단에 연결함