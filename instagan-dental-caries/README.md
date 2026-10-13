# InstaGAN 기반 치아 위치별 충치 이미지 생성

정상 치아 이미지로부터 충치 치아 이미지를 생성하기 위해 InstaGAN을 참고해 구성한 수업 프로젝트입니다. 치아 위치별 이미지와 분할 마스크를 전처리하고, 마스크를 사용하는 GAN 학습·평가 코드를 작성했습니다.

## 주요 작업

- 상·하악 24개 치아 위치별 정상·충치 이미지 및 마스크 구성
- 분할 polygon으로부터 마스크 생성 및 치아 영역 추출
- 같은 위치의 정상·충치 이미지를 비대응 방식으로 선택하는 데이터셋 구현
- RGB와 마스크를 입력하는 Generator 및 PatchGAN Discriminator 구현
- adversarial·L1·masked L1·VGG perceptual loss 기반 학습
- 위치별 체크포인트, 생성 이미지, 학습 이력 저장
- FID·Inception Score·LPIPS 평가 코드 작성

## 데이터와 모델

- 입력: 128×128 RGB 치아 이미지와 분할 마스크
- 데이터: 정상·충치 치아 이미지와 치아 위치·충치 여부를 포함한 어노테이션
- 기술: Python, PyTorch, torchvision, OpenCV, LPIPS

## 파일 및 실행

| 파일 | 내용 |
| --- | --- |
| [instaGAN_ver1.3.py](instaGAN_ver1.3.py) | 데이터 처리, 모델, 학습 및 평가 |
| [requirements.txt](requirements.txt) | 소스 import 기준 설치 목록 |
| [docs/SETUP.md](docs/SETUP.md) | 데이터 형식·실행 방법·출력 위치 |
| [docs/CONFIGURATION_CHANGES.diff](docs/CONFIGURATION_CHANGES.diff) | 공개용 경로·GPU 설정 변경 3줄 |

실제 데이터와 학습된 가중치는 포함하지 않았습니다. InstaGAN을 참고해 치아 데이터에 맞게 구성한 구현이며, 이미지 단위로 학습·검증을 분할합니다. LPIPS는 정상 입력과 생성 이미지 간에 계산합니다. 최종 성능 수치는 포함하지 않았습니다.
