import cv2
import numpy as np

# 배경 이미지 불러오기
background = cv2.imread("hogwart.jpg")

# 크로마키 적용할 영상 불러오기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # 배경 크기와 맞추기
    background_resized = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    # HSV 색 공간으로 변환 (녹색 검출을 쉽게 하기 위함)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 녹색 범위 정의 (값은 상황에 맞게 조정 가능)
    lower_green = np.array([35, 80, 80])
    upper_green = np.array([85, 255, 255])

    # 마스크 생성 (녹색 부분만 검출)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 마스크 반전 (사람/물체 부분만 남김)
    mask_inv = cv2.bitwise_not(mask)

    # 사람/물체 부분 추출
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # 배경 부분 추출
    bg = cv2.bitwise_and(background_resized, background_resized, mask=mask)

    # 합성
    result = cv2.add(fg, bg)

    cv2.imshow("Chroma Key", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
