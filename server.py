import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import flask
import base64
import tempfile
import traceback
from flask import Flask, Response, stream_with_context
from inference_vision import OmniVisionInference


class OmniChatServer(object):
    def __init__(self, ip='0.0.0.0', port=60808, run_app=True,
                 ckpt_dir='./checkpoint', device='cuda:0') -> None:
        server = Flask(__name__)
        # CORS(server, resources=r"/*")
        # server.config["JSON_AS_ASCII"] = False

        self.client = OmniVisionInference(ckpt_dir, device)
        self.client.warm_up()

        server.route("/chat", methods=["POST"])(self.chat)

        if run_app:
            server.run(host=ip, port=port, threaded=False)
        else:
            self.server = server

    def chat(self) -> Response:

        req_data = flask.request.get_json()
        try:
            audio_data_buf = req_data["audio"].encode("utf-8")
            audio_data_buf = base64.b64decode(audio_data_buf)
            stream_stride = req_data.get("stream_stride", 4)
            max_tokens = req_data.get("max_tokens", 2048)

            image_data_buf = req_data.get("image", None)
            if image_data_buf:
                image_data_buf = image_data_buf.encode("utf-8")
                image_data_buf = base64.b64decode(image_data_buf)

            audio_path, img_path = None, None
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_f, \
                 tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_f:
                audio_f.write(audio_data_buf)
                audio_path = audio_f.name

                if image_data_buf:
                    img_f.write(image_data_buf)
                    img_path = img_f.name
                else:
                    img_path = None

                if img_path is not None:
                    resp_generator = self.client.run_vision_AA_batch_stream(audio_f.name, img_f.name,
                                                                             stream_stride, max_tokens,
                                                                             save_path='./vision_qa_out_cache.wav')
                else:
                    resp_generator = self.client.run_AT_batch_stream(audio_f.name, stream_stride,
                                                                      max_tokens,
                                                                      save_path='./audio_qa_out_cache.wav')
                return Response(stream_with_context(self.generator(resp_generator)),
                                mimetype='multipart/x-mixed-replace; boundary=frame')
        except Exception as e:
            print(traceback.format_exc())
            return Response("An error occurred", status=500)

    def generator(self, resp_generator):
        for audio_stream, text_stream in resp_generator:
            yield b'\r\n--frame\r\n'
            yield b'Content-Type: audio/wav\r\n\r\n'
            yield audio_stream
            yield b'\r\n--frame\r\n'
            yield b'Content-Type: text/plain\r\n\r\n'
            yield text_stream.encode()


# CUDA_VISIBLE_DEVICES=1 gunicorn -w 2 -b 0.0.0.0:60808 'server:create_app()'
def create_app():
    server = OmniChatServer(run_app=False)
    return server.server


def serve(ip='0.0.0.0', port=60808, device='cuda:0'):

    OmniChatServer(ip, port=port,run_app=True, device=device)


if __name__ == "__main__":
    import fire
    fire.Fire(serve)

