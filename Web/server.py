from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.templating import Jinja2Templates
import asyncio, cv2, time, threading

import opencv_worker   # ✅ YOLO 모듈
shared = {"camera_frame": None, "lock": threading.Lock()}

app = FastAPI()
templates = Jinja2Templates(directory="templates")

connected_clients = set()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/stt")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    print(f"[SERVER] 연결됨: {ws.client}")
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=30)
                for client in list(connected_clients):
                    try:
                        await client.send_text(msg)
                    except:
                        connected_clients.discard(client)
            except asyncio.TimeoutError:
                for client in list(connected_clients):
                    try:
                        await client.send_text("ping")
                    except:
                        connected_clients.discard(client)
    finally:
        connected_clients.discard(ws)
        if not ws.client_state.name == "DISCONNECTED":
            await ws.close()
        print(f"[SERVER] 연결 해제: {ws.client}")

def gen_frames():
    while True:
        with shared["lock"]:
            if shared["camera_frame"] is None:
                time.sleep(0.03)
                continue
            frame_copy = shared["camera_frame"].copy()
        ret, buffer = cv2.imencode('.jpg', frame_copy)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
def video_feed():
    response = StreamingResponse(gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")
    return response

@app.get("/video_frame")
def video_frame():
    with shared["lock"]:
        if shared["camera_frame"] is None:
            return Response(status_code=204)
        ret, buffer = cv2.imencode('.jpg', shared["camera_frame"])
        if not ret:
            return Response(status_code=500)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.on_event("startup")
def startup_event():
    threading.Thread(
        target=opencv_worker.run_cam_multithread,
        kwargs={"shared": shared},
        daemon=True
    ).start()
    print("[SERVER] YOLO worker thread started")
