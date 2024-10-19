import http.server
import socketserver
import socketio
import threading
import eventlet
import webbrowser
import time

# Socket.IOのサーバーを作成
sio = socketio.Server(cors_allowed_origins='*')
# sio = socketio.Server(cors_allowed_origins='*')


# ソケット接続時の処理
@sio.event
def connect(sid, environ):
    print(f"User {sid} connected")  # 接続されたユーザーのIDを表示

# シグナリング処理
@sio.event
def signal(sid, data):
    print(f"Received signal from {sid}: {data}")  # 受信したシグナリングデータを表示
    sio.emit('signal', data, skip_sid=sid)  # 他のクライアントにデータを送信

# 切断時の処理
@sio.event
def disconnect(sid):
    print(f"User {sid} disconnected")  # 切断されたユーザーのIDを表示

# HTTPリクエストを処理するハンドラー
class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)  # HTTPステータス200を返す
            self.send_header('Content-type', 'text/html')  # レスポンスの内容タイプをHTMLに設定
            self.end_headers()
            with open('index.html', 'rb') as f:
                self.wfile.write(f.read())  # index.htmlの内容をレスポンスとして返す
        else:
            super().do_GET()  # その他のリクエストは親クラスの処理を行う

# HTTPサーバーの開始
def start_server():
    PORT = 8000  # ポート番号の設定
    handler = CustomHandler  # ハンドラーの指定
    httpd = socketserver.TCPServer(("", PORT), handler)  # TCPサーバーを作成
    print(f"Serving HTTP on port {PORT}...")  # サーバーが起動したことを表示
    httpd.serve_forever()  # 永久にリクエストを待ち続ける

# Socket.IOサーバーの開始
def start_socket_server():
    app = socketio.WSGIApp(sio)  # Socket.IOアプリを作成
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)  # Socket.IOサーバーを起動

# メインスレッドでHTTPサーバーを、別スレッドでSocket.IOサーバーを実行
if __name__ == "__main__":
    # HTMLファイルの作成
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>簡単なビデオ通話アプリ</title>
  <style>
    video {
      width: 400px;  /* ビデオの幅 */
      height: 300px; /* ビデオの高さ */
      border: 1px solid black; /* ビデオの境界線 */
    }
  </style>
</head>
<body>
  <h1>通話相手を選択してください</h1>
  <button id="callButton">通話を開始</button> <!-- 通話を開始するボタン -->
  <video id="localVideo" autoplay muted></video> <!-- 自分のビデオ -->
  <video id="remoteVideo" autoplay></video> <!-- 相手のビデオ -->
  <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script> <!-- Socket.IOライブラリの読み込み -->
  <script>
    const socket = io('http://localhost:5000');  // Socket.IOサーバーのポートを指定
    let localConnection; // RTCPeerConnectionオブジェクト
    let localVideo = document.getElementById('localVideo'); // 自分のビデオ要素
    let remoteVideo = document.getElementById('remoteVideo'); // 相手のビデオ要素

    // ICEサーバーの設定
    const servers = {
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] // STUNサーバーのURL
    };

    // ボタンを押したときの動作
    document.getElementById('callButton').addEventListener('click', async () => {
      localConnection = new RTCPeerConnection(servers); // 新しいRTCPeerConnectionオブジェクトを作成
      localConnection.onicecandidate = event => { // ICE候補を受信したときの処理
        if (event.candidate) {
          socket.emit('signal', { candidate: event.candidate }); // ICE候補をサーバーに送信
        }
      };

      localConnection.ontrack = event => { // リモートトラックを受信したときの処理
        remoteVideo.srcObject = event.streams[0]; // リモートビデオにストリームを設定
      };

      // メディアデバイスから音声とビデオのストリームを取得
      let stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      stream.getTracks().forEach(track => localConnection.addTrack(track, stream)); // ストリームのトラックをローカル接続に追加
      localVideo.srcObject = stream; // 自分のビデオにストリームを設定

      let offer = await localConnection.createOffer(); // オファーを作成
      await localConnection.setLocalDescription(offer); // ローカル接続にオファーを設定
      socket.emit('signal', { sdp: offer }); // オファーをサーバーに送信
    });

    // シグナリング情報を受信したとき
    socket.on('signal', async data => {
      if (data.sdp) { // SDP情報がある場合
        await localConnection.setRemoteDescription(new RTCSessionDescription(data.sdp)); // リモートSDPを設定
        if (data.sdp.type === 'offer') { // 受信したSDPがオファーの場合
          let answer = await localConnection.createAnswer(); // アンサーを作成
          await localConnection.setLocalDescription(answer); // ローカル接続にアンサーを設定
          socket.emit('signal', { sdp: answer }); // アンサーをサーバーに送信
        }
      } else if (data.candidate) { // ICE候補がある場合
        await localConnection.addIceCandidate(new RTCIceCandidate(data.candidate)); // ICE候補をローカル接続に追加
      }
    });
  </script>
</body>
</html>
        """)

    # サーバーを別スレッドで起動
    http_thread = threading.Thread(target=start_server)
    http_thread.daemon = True
    http_thread.start()  # HTTPサーバーを非同期で実行

    # Socket.IOサーバーをメインスレッドで起動
    start_socket_server()

    # サーバーが起動するまで待機
    time.sleep(1)  # サーバーが起動するのを少し待つ

    # 2つのブラウザウィンドウを自動で起動
    webbrowser.open("http://localhost:8000")  # 1つ目のブラウザを開く
    webbrowser.open("http://localhost:8000")  # 2つ目のブラウザを開く