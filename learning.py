import pandas as pd  # 데이터 분석과 조작을 도와주는 라이브러리로, 주로 표 형태의 데이터를 쉽게 다룰 수 있게 해줍니다.
import numpy as np  # 숫자 계산을 효율적으로 할 수 있게 도와주는 라이브러리로, 특히 배열(리스트 같은 것)을 다루는 데 강력합니다.
import tensorflow as tf  # 머신러닝과 딥러닝 모델을 만들고 학습시키는 데 사용하는 라이브러리입니다.
from sklearn.model_selection import train_test_split  # 데이터를 학습용과 테스트용으로 나누는 데 사용하는 함수입니다

# 모델이 너무 과적합(overfitting)되지 않도록, 즉 학습 데이터에만 너무 치우치지 않도록 학습을 조기 종료하는 기능을 제공하는 콜백 함수입니다.
EarlyStopping = tf.keras.callbacks.EarlyStopping

def load_and_preprocess_data(file_path, sample_size=1.0):
    """ 
    주어진 파일에서 데이터를 불러오고, 이미지와 레이블을 전처리하는 함수입니다.
    
    Parameters:
    - file_path: 데이터를 불러올 CSV 파일의 경로
    - sample_size: 데이터를 얼마나 샘플링할 것인지 비율(0.7이면 70%만 사용)을 지정합니다.
    
    Returns:
    - 데이터를 학습용과 테스트용으로 나눈 후 반환합니다.
    """
    
    data = pd.read_csv(file_path)  # CSV 파일에서 데이터를 읽어옵니다. 읽어온 데이터는 표 형태(데이터프레임)로 저장됩니다.
    pixels = data['pixels'].tolist()  # 'pixels' 열에 있는 데이터를 리스트로 변환합니다. 여기에는 이미지의 픽셀 데이터가 문자열 형태로 들어 있습니다.

    # 각 이미지의 픽셀 데이터를 48x48x1(가로 48, 세로 48, 흑백 1 채널)의 배열로 변환합니다.
    faces = np.array([np.array(pixel_sequence.split(), dtype=np.uint8).reshape(48, 48, 1) for pixel_sequence in pixels])
    faces = faces.astype('float32') / 255.0  # 픽셀 값을 0에서 1 사이의 값으로 정규화합니다. 이렇게 하면 모델이 학습하기 더 쉬워집니다.
    
    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    # csv 파일에서 'emotion' 열에는 감정(예: 0 1 2)이 기록되어 있습니다. 이 감정을 원-핫 인코딩하여 이진 벡터로 변환합니다.
    emotions = pd.get_dummies(data['emotion']).values  

    # 데이터의 양을 줄이고 싶을 때 사용합니다. 예를 들어, sample_size가 0.7이면 전체 데이터의 70%만 사용합니다.
    if sample_size < 1.0:
        faces, _, emotions, _ = train_test_split(faces, emotions, train_size=sample_size, random_state=42)

    # 데이터를 학습용과 테스트용으로 나눕니다. 보통 80%는 학습용, 20%는 테스트용으로 사용합니다.
    return train_test_split(faces, emotions, test_size=0.2, random_state=42)

def build_model(input_shape):
    """ 
    Convolutional Neural Network (CNN) 모델을 설계하는 함수입니다.
    
    Parameters:
    - input_shape: 입력 데이터의 형태(예: 48x48 크기의 이미지와 1개의 색상 채널)입니다.
    
    Returns:
    - 설계한 CNN 모델을 반환합니다.
    """
    
    model = tf.keras.Sequential([  # 순차적으로 모델의 층(layer)을 쌓아가는 방식으로 모델을 구성합니다.
        # 첫 번째 합성곱(Convolution) 레이어입니다.
        # 32개의 필터를 사용하여 이미지에서 특징을 추출합니다. 이 과정에서 ReLU라는 활성화 함수가 적용됩니다.
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),  
        tf.keras.layers.BatchNormalization(),  # 배치 정규화(Batch Normalization)로 학습을 안정화시킵니다.
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # 풀링(Pooling) 레이어로 이미지의 크기를 줄여줍니다.

        # 두 번째 합성곱 레이어입니다.
        # 이번에는 64개의 필터를 사용하여 더 깊은 특징을 추출합니다.
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),  
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # 다시 풀링 레이어로 이미지 크기를 줄입니다.

        tf.keras.layers.Flatten(),  # 2D로 이루어진 이미지 데이터를 1D로 평평하게 만듭니다. 완전 연결 레이어에 입력하기 위해서입니다.
        tf.keras.layers.Dropout(0.3),  # 드롭아웃(Dropout)을 적용하여 과적합을 방지합니다.
        tf.keras.layers.Dense(128, activation='relu'),  # 128개의 뉴런을 가진 완전 연결(Dense) 레이어입니다.
        tf.keras.layers.Dropout(0.5),  # 추가 드롭아웃을 적용하여 과적합을 더 방지합니다.
        tf.keras.layers.Dense(7, activation='softmax')  # 마지막 출력 레이어입니다. 7개의 감정 클래스 각각에 대해 확률을 출력합니다.
    ])
    return model

def train_and_save_model(file_path, sample_size, epochs, model_name):
    """
    데이터를 로드하고 전처리한 후 모델을 학습시키고 저장하는 함수입니다.
    
    Parameters:
    - file_path: CSV 데이터 파일 경로
    - sample_size: 데이터 샘플 크기 비율
    - epochs: 학습할 에포크 수
    - model_name: 저장할 모델 파일 이름
    """
    # 데이터를 불러오고 전처리합니다.
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, sample_size=sample_size)
    
    # 모델을 정의합니다. 이 모델은 (48, 48, 1) 형태의 흑백 이미지를 입력받습니다.
    model = build_model((48, 48, 1))
    
    # 모델을 컴파일합니다. 학습을 위해 Adam이라는 최적화 알고리즘을 사용하고,
    # 다중 클래스 분류를 위해 'categorical_crossentropy'라는 손실 함수를 사용합니다.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  
    
    # EarlyStopping 콜백을 설정합니다.
    # 검증 손실이 개선되지 않으면 학습을 중단하고, 가장 좋은 가중치(모델 상태)를 복원합니다.
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # 모델을 학습시킵니다.
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, callbacks=[early_stopping])  
    
    # 모델 학습 후 h5 파일로 저장
    model.save(f'{model_name}.h5')
    print(f'Model saved as {model_name}.h5')

# 학습 설정: (에포크, 데이터 샘플 크기, 모델 이름)
configurations = [
    (50, 1.0, 'emotion_recognition_model_epochs50_data100'),
    (50, 0.5, 'emotion_recognition_model_epochs50_data50'),
    (25, 1.0, 'emotion_recognition_model_epochs25_data100'),
    (25, 0.5, 'emotion_recognition_model_epochs25_data50')
]

file_path = 'fer2013.csv'  # 데이터 파일 경로

for epochs, sample_size, model_name in configurations:
    print(f'\nTraining model: {model_name} with epochs={epochs} and sample_size={sample_size}')
    train_and_save_model(file_path, sample_size, epochs, model_name)
