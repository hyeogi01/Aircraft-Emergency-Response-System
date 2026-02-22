import threading

# YOLO → 서버가 공유할 전역 변수
camera_frame = None
lock = threading.Lock()
