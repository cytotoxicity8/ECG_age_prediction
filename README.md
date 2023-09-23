# 서울대병원 심전도 기반 나이 예측 
팀 DSHS 연합의 추론 코드입니다.

팀장 이메일: cytotoxicity8@kaist.ac.kr

## 컴퓨터 사양 및 실행 시간
이 일련의 과정은 제공된 valid data에서 한 번 테스트를 수행했습니다.

램 32GB, VRAM 12GB (RTX 3060)으로 1시간 이내로 실행되었습니다.

## 개발환경

pytorch 2.0 도커 이미지를 활용합니다.

    docker pull pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

저희 팀에서는 다음 명령어로 이미지를 run하였습니다.

    docker run -itd --rm -p 8888:8888 -v /(our path):/root/share --gpus all --name ecg_docker --ipc=host  pytorch/pytorch:standard-2.0.0

다만, pytorch 2.0.0이 설치된 환경이면 큰 문제 없이 실행될 것으로 예상합니다.

docker container 안에서 requirements.txt의 패키지를 설치했습니다.
이 설치 코드는 향후 실행할 주피터 노트북에도 포함되어 있습니다.
tsai, sktime, joblib이 잘 설치되어야 합니다.

    pip install -r requirements.txt

Juypter notebook 활용을 위한 세팅이 필요할 수 있습니다.

## 데이터 파일
기본적으로 모든 데이터는 data 폴더 안에 저장되어야 합니다. 만약 폴더 이름이 다른 경우, config.py 파일 내부를 수정해주셔야 합니다. 자세한 건 밑에서 설명드리겠습니다.

**심전도 데이터는 반드시 adult와 child가 분리된 .zip 형태여야 합니다.**
드라이브에서 제공해주신 파일인, ecg_adult_numpy_valid.zip 형태를 input으로 받고 있습니다. 압축이 풀린 상태나, .egg 파일은 현재 활용하기 어렵습니다. 데스크탑 램이 크지 않아 압축 파일을 그대로 활용하였습니다. .zip 활용이 불가능하다면 연락 부탁드립니다.

## 모델 파일
weights 폴더에, 구글 폼에 제출한 압축 파일을 해제해주시면 됩니다.

## 재현 폴더 구조

```
📦ECG_age_prediction
 ┣ 📂data
 ┃ ┣ 📜ECG_adult_numpy_valid.zip
 ┃ ┣ 📜ECG_child_numpy_valid.zip
 ┃ ┗ 📜submission.csv
 ┣ 📂outputs
 ┃ ┣ 📜submission.csv
 ┃ ┗ 📜submission_check.csv
 ┣ 📂weights
 ┃ ┣ 📂adult
 ┃ ┃ ┣ 📜inception1_39_model.pth
 ┃ ┃ ┣ 📜inception2_39_model.pth
 ┃ ┃ ┣ 📜resnet1_39_model.pth
 ┃ ┃ ┣ 📜resnet2_39_model.pth
 ┃ ┃ ┣ 📜rocket1_features10000_dilation32_ensemble5.pkl
 ┃ ┃ ┗ 📜rocket2_features12000_dilation36_ensemble3.pkl
 ┃ ┗ 📂child
 ┃ ┃ ┣ 📜inception_child_79_model.pth
 ┃ ┃ ┣ 📜resnet_child_69_model.pth
 ┃ ┃ ┗ 📜rocket_child_ensemble5.pkl
 ┣ 📜README.md
 ┣ 📜config.py
 ┣ 📜dataset.py
 ┣ 📜inference.ipynb
 ┣ 📜requirements.txt
 ┣ 📜resnet.py
 ┗ 📜utils.py
```

## config.py

data 경로와 파일 이름 등 몇 가지 정보를 수정하려면 config.py를 활용해주세요. 
```
DATA_PATH = "data" #데이터 디렉토리
CHILD_ECG_FILENAME = "ECG_child_numpy_valid.zip" #child의 ECG file 이름
ADULT_ECG_FILENAME = "ECG_adult_numpy_valid.zip" #adult의 ECG file 이름

INFO_FILENAME = "submission.csv" #submission file 이름
OUTPUT_PATH = "outputs" #추론 결과 디렉토리

DEVICE = "cuda:0" #사용할 GPU
BATCH_SIZE = 256 #배치 사이즈 (크게 중요한 요소는 아닙니다.)
```

## inference.ipynb

config.py 세팅 이후, inference.ipynb를 "Run All"하면 모든 추론 과정이 자동으로 실행됩니다.

**outputs/submission.csv 파일을 활용해주시면 됩니다.**

***outputs/submission_check.csv는 private scoring에 활용되지 않습니다.*** 혹시 모를 디버깅용 파일이니 무시하셔도 좋습니다. (앙상블에 활용된 모든 모델의 예측 결과를 담고 있습니다.)
