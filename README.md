# 🧠 AI 얼굴 및 감정 인식 시스템

이 프로젝트는 딥러닝 기반 Convolutional Neural Network(CNN)를 이용하여 이미지 속 인물의 얼굴을 감지하고, 감정을 인식하는 Python 프로그램입니다.

<p align="center">
  <img src="output_emotion_recognition_model_epochs50_data100.jpg" alt="Emotion Recognition Example" width="500">
</p>

---

## 📌 주요 기능

- 얼굴 감지: OpenCV의 Haar Cascade를 이용한 얼굴 감지
- 감정 분류: CNN 모델을 활용하여 7가지 감정 예측
  - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- 모델 학습: FER-2013 데이터셋 기반 모델 학습
- 다중 모델 예측 결과 비교 지원

---

## 🛠 사용 기술

- Python 3
- TensorFlow / Keras
- OpenCV
- Pandas / NumPy
- scikit-learn

---

## 📁 파일 구조

```
📂 프로젝트 루트
├── start.py                   # 이미지에서 얼굴 및 감정 인식 실행 파일
├── learning.py                # CNN 모델 학습 및 저장 스크립트
├── fer2013.csv                # 학습용 데이터셋 (FER-2013)
├── *.h5                       # 학습된 감정 인식 모델들
└── output_*.jpg               # 예측 결과 이미지
```

---

## 📥 설치 방법

```bash
# 가상 환경 설정 권장
python -m venv venv
source venv/bin/activate   # (Windows는 venv\Scripts\activate)

# 필수 라이브러리 설치
pip install tensorflow opencv-python pandas numpy scikit-learn
```

---

## 🧪 모델 학습 방법

`fer2013.csv` 파일을 동일한 디렉토리에 위치시킨 후 다음 명령어 실행:

```bash
python learning.py
```

4개의 모델이 학습되며 `.h5` 파일로 저장됩니다:
- `emotion_recognition_model_epochs50_data100.h5`
- `emotion_recognition_model_epochs50_data50.h5`
- `emotion_recognition_model_epochs25_data100.h5`
- `emotion_recognition_model_epochs25_data50.h5`

---

## 🎯 감정 인식 실행 방법

아래 명령어를 실행하면 입력 이미지에서 얼굴을 감지하고 감정을 예측하여 이미지로 저장합니다.

```bash
python start.py <이미지경로> [결과파일_접두사]
```

예시:

```bash
python start.py test.jpg result
```

결과:
- `result_emotion_recognition_model_epochs50_data100.jpg`
- `result_emotion_recognition_model_epochs50_data50.jpg`
- ...

---

## 📊 예측 감정 클래스

| Label | 감정 (Emotion) |
|-------|----------------|
| 0     | Angry          |
| 1     | Disgust        |
| 2     | Fear           |
| 3     | Happy          |
| 4     | Sad            |
| 5     | Surprise       |
| 6     | Neutral        |

---

## 💡 참고 사항

- FER2013 데이터셋은 Kaggle에서 다운로드할 수 있습니다: https://www.kaggle.com/datasets/msambare/fer2013
- 입력 이미지는 얼굴이 명확하게 보이는 정면 사진일수록 정확도가 높습니다.

---
