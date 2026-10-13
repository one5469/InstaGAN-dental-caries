# 데이터 준비와 실행

## 환경

저장소 루트에서 `python -m pip install -r requirements.txt`로 필요한 패키지를 설치합니다. 설치 목록은 소스의 import를 기준으로 작성했으며, 원래 실험 환경의 고정 버전 목록은 아닙니다. PyTorch·torchvision의 GPU 환경은 사용하시는 CUDA 환경에 맞춰야 합니다.

VGG·Inception·LPIPS의 사전학습 가중치는 처음 사용할 때 별도 다운로드가 필요할 수 있습니다.

## 데이터

`INSTAGAN_DATA_DIR`은 다음 항목을 포함하는 기존 데이터 디렉터리를 지정합니다. 데이터 자체는 저장소에 포함하지 않습니다.

| 데이터 루트 아래 경로 | 용도 |
| --- | --- |
| `metadata_v20_1_20251123_Validation_data.json` | 기존 메타데이터 경로 설정 |
| `Validation Data/라벨데이터/lower/5.lower/` | 하악 JSON 어노테이션 |
| `Validation Data/라벨데이터/upper/4.upper/` | 상악 JSON 어노테이션 |
| `Validation Data/원데이터/lower/5.lower/` | 하악 PNG 이미지 |
| `Validation Data/원데이터/upper/4.upper/` | 상악 PNG 이미지 |

JSON과 PNG의 파일 stem이 일치해야 합니다. 어노테이션의 `tooth` 항목에는 `teeth_num`, `segmentation`, `decayed`를 사용합니다. 한 이미지에서 해당 상악 또는 하악의 12개 치아 위치가 모두 있는지를 확인한 후 데이터를 구성합니다. 정확한 데이터 출처와 배포 정보는 별도 확인 후 README에 추가할 수 있습니다.

## 실행

Linux/bash 예시입니다. 저장소 루트에서 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 INSTAGAN_DATA_DIR=/absolute/path/to/teeth_analysis INSTAGAN_OUTPUT_DIR=./outputs python instaGAN_ver1.3.py
```

원본의 `GPU_ID=3`은 유지했지만, 이제 외부에서 지정한 `CUDA_VISIBLE_DEVICES`를 덮어쓰지 않습니다. 지정하지 않으면 원본처럼 3번 GPU를 요청합니다. 데이터 경로의 기본값은 `./data/teeth_analysis`, 출력 경로의 기본값은 `./outputs`입니다.

`TOOTH_SIZE=128`, `BATCH_SIZE=8`, `NUM_EPOCHS=500`, 학습률 및 loss·모델 코드는 원본 그대로입니다. 실행하면 상·하악 위치별 학습을 진행하므로 확인용 짧은 실행이 필요할 때는 원본 설정 블록의 `NUM_EPOCHS` 등을 직접 조정하세요.

출력 디렉터리에 `checkpoints/`, `results/`, `debug/`, `metrics/`가 생성됩니다. 정상 또는 충치 데이터가 없는 위치는 건너뜁니다. 데이터가 제공되지 않아 이번 정리에서 전체 학습·생성 실행은 검증하지 않았습니다.
