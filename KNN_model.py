import os
import pickle
import face_recognition
from sklearn import neighbors
from picamera import PiCamera
from time import sleep
import numpy as np
from picamera.array import PiRGBArray
import RPi.GPIO as GPIO

# GPIO 핀 설정
RED_PIN = 18
YELLOW_PIN = 12
GREEN_PIN = 16
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(YELLOW_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)

def predict_single_image(image_path, model_path, distance_threshold=0.4):
    """
    Predicts the person in a single image using a pre-trained KNN model.

    :param image_path: Path to the test image.
    :param model_path: Path to the trained KNN model.
    :param distance_threshold: Threshold for face classification. Default is 0.6.
    :return: A string indicating the name of the person or 'UNKNOWN'.
    """
    if not os.path.isfile(image_path):
        raise ValueError(f"Image file not found: {image_path}")

    if not os.path.isfile(model_path):
        raise ValueError(f"KNN model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return "No face detected"

    face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
    are_matches = [dist[0] <= distance_threshold for dist in closest_distances[0]]

    results = [pred if match else "UNKNOWN"
               for pred, match in zip(knn_clf.predict(face_encodings), are_matches)]

    return results[0] if results else "UNKNOWN"

def capture_image(capture_path="./test/capture.jpg"):
    """
    Capture an image using PiCamera and save it to the specified path.

    :param capture_path: Path to save the captured image.
    """
    print("Capturing image...")
    
    # PiCamera 설정
    camera = PiCamera()
    camera.resolution = (1024, 768)  # 해상도 설정
    output = np.empty((768, 1024, 3), dtype=np.uint8)  # NumPy 배열로 이미지 출력을 설정
    
    # PiRGBArray로 카메라 출력을 캡처합니다.
    raw_capture = PiRGBArray(camera, size=(1024, 768))
    
    # 카메라 프리뷰 시작 및 잠시 대기
    camera.start_preview()
    sleep(2)  # 카메라가 적응할 시간을 줍니다.
    
    # 카메라로 이미지 캡처
    camera.capture(raw_capture, format="rgb")
    image = raw_capture.array  # 이미지를 NumPy 배열로 가져옵니다.
    
    # 이미지를 저장 (이 부분은 NumPy 배열을 이미지 파일로 저장하는 코드입니다)
    from PIL import Image
    pil_image = Image.fromarray(image)
    pil_image.save(capture_path)
    
    # 카메라 종료
    camera.stop_preview()
    camera.close()
    
    print(f"Image captured and saved to {capture_path}")

def control_led(result):
    """
    Controls the LED based on the face recognition result.

    :param result: The result of the face recognition (e.g., "GUN", "LIM", "UNKNOWN", "No face detected").
    """
    if result == "GUN":
        GPIO.output(YELLOW_PIN, GPIO.HIGH)  # YELLOW LED 켜기
        print("YELLOW LED ON (GUN)")
        sleep(5)  # 5초 동안 YELLOW LED 점등
        GPIO.output(YELLOW_PIN, GPIO.LOW)  # YELLOW LED 끄기
        print("YELLOW LED OFF")
    elif result == "LIM":
        GPIO.output(GREEN_PIN, GPIO.HIGH)  # GREEN LED 켜기
        print("GREEN LED ON (LIM)")
        sleep(5)  # 5초 동안 GREEN LED 점등
        GPIO.output(GREEN_PIN, GPIO.LOW)  # GREEN LED 끄기
        print("GREEN LED OFF")
    elif result == "UNKNOWN":
        GPIO.output(RED_PIN, GPIO.HIGH)  # RED LED 켜기
        print("RED LED ON (UNKNOWN)")
        sleep(5)  # 5초 동안 RED LED 점등
        GPIO.output(RED_PIN, GPIO.LOW)  # RED LED 끄기
        print("RED LED OFF")
    else:
        # 얼굴이 감지되지 않으면 아무 LED도 켜지지 않음
        print("No face detected, no LED on.")

def predict_images_in_test(model_path, test_dir="./test"):
    """
    Predict all images in the test directory using a pre-trained KNN model.

    :param model_path: Path to the trained KNN model.
    :param test_dir: Directory containing test images.
    """
    print("Predicting images in the test directory...")
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not test_images:
        print("No test images found.")
        return

    for image_file in test_images:
        full_path = os.path.join(test_dir, image_file)
        print(f"Processing {image_file}...")
        result = predict_single_image(full_path, model_path)
        print(f"- Prediction result for {image_file}: {result}")

        # LED 제어 함수 호출
        control_led(result)


if __name__ == "__main__":
    # Paths
    test_dir = "./test"
    model_path = "./model/trained_knn_model.clf"
    capture_path = os.path.join(test_dir, "capture.jpg")

    try:
        while True:  # 무한 반복문으로 계속 실행
            # Step 1: Capture a new image
            capture_image(capture_path)

            # Step 2: Predict the captured image and other test images
            predict_images_in_test(model_path, test_dir)

    except KeyboardInterrupt:
        print("Program interrupted, cleaning up GPIO...")
        GPIO.cleanup()  # 프로그램 종료 시 GPIO 정리
