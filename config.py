DATA_PATH = "data" #데이터 디렉토리
CHILD_ECG_FILENAME = "ECG_child_numpy_valid.zip" #child의 ECG file 이름
ADULT_ECG_FILENAME = "ECG_adult_numpy_valid.zip" #adult의 ECG file 이름

INFO_FILENAME = "submission.csv" #submission file 이름
OUTPUT_PATH = "outputs" #추론 결과 디렉토리

DEVICE = "cuda:0" #사용할 GPU
BATCH_SIZE = 256 #배치 사이즈 (크게 중요한 요소는 아닙니다.)