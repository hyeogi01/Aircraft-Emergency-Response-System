import socket

HOST = "127.0.0.1"
PORT = 5000

def start_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 포트 재사용 옵션 추가 (TIME_WAIT 문제 방지)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        s.bind((HOST, PORT))
        s.listen()
        print(f"🚀 Server started on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            with conn:
                print("Connected by", addr)
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    print("📩 Received:", data.decode())
                    conn.sendall(b"Message received")

    except KeyboardInterrupt:
        print("\n🛑 Server shutting down...")

    finally:
        s.close()
        print("🔌 Socket closed, port released.")

if __name__ == "__main__":
    start_server()
