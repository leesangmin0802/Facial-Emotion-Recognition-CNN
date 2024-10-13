import cv2
import numpy as np
import tensorflow as tf
import sys
import os

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"모델 '{model_path}'이 성공적으로 로드되었습니다.")
        return model
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        sys.exit()

def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        sys.exit()
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드하는 데 실패했습니다: {image_path}")
        sys.exit()
    return image

def detect_faces(gray_frame, face_cascade):
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces

def preprocess_face(face_img):
    resized_frame = cv2.resize(face_img, (48, 48))
    reshaped_frame = resized_frame.reshape(1, 48, 48, 1)
    normalized_frame = reshaped_frame.astype('float32') / 255.0
    return normalized_frame

def process_image(image_path, models, face_cascade, emotion_labels, output_prefix):
    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, face_cascade)

    if len(faces) == 0:
        print("얼굴을 감지하지 못했습니다.")
    else:
        for i, (model, model_name) in enumerate(models):
            # 각 모델에 대해 새로운 이미지 복사본을 생성
            output_image = image.copy()
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                processed_face = preprocess_face(face_img)

                # 모델 예측
                prediction = model.predict(processed_face)
                predicted_emotion = np.argmax(prediction)
                emotion = emotion_labels[predicted_emotion]
                confidence = prediction[0][predicted_emotion]

                # 얼굴에 사각형 그리기
                cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # 감정과 신뢰도 텍스트 표시
                cv2.putText(output_image, f"{emotion} ({confidence*100:.1f}%)", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (36,255,12), 2)

            # 결과 이미지 저장
            output_image_path = f"{output_prefix}_{model_name.split('.')[0]}.jpg"
            cv2.imwrite(output_image_path, output_image)
            print(f"결과 이미지가 저장되었습니다: {output_image_path}")


def main():
    if len(sys.argv) < 2:
        print("사용법: python emotion_recognition_image.py <이미지_경로> [출력_접두사]")
        sys.exit()

    image_path = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "output"

    # 모델과 얼굴 검출기 초기화
    model_names = [
        'emotion_recognition_model_epochs50_data100.h5',
        'emotion_recognition_model_epochs50_data50.h5',
        'emotion_recognition_model_epochs25_data100.h5',
        'emotion_recognition_model_epochs25_data50.h5'
    ]
    
    models = [(load_model(model_name), model_name.split('.')[0]) for model_name in model_names]
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 감정 레이블 설정
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # 이미지 처리
    process_image(image_path, models, face_cascade, emotion_labels, output_prefix)

if __name__ == "__main__":
    main()
