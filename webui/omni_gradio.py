"""A simple web interactive chat demo based on gradio."""

import os
import gradio as gr
from gradio_webrtc import WebRTC, AdditionalOutputs, ReplyOnPause
import base64
import numpy as np
import requests
import io
from pydub import AudioSegment


API_URL = os.getenv("API_URL", None)
client = None

if API_URL is None:
    from inference import OmniInference
    omni_client = OmniInference('./checkpoint', 'cuda:0')
    omni_client.warm_up()


OUT_CHUNK = 4096
OUT_RATE = 24000
OUT_CHANNELS = 1


# Only needed if deploying on cloud provider
account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

if account_sid and auth_token:
    from twilio.rest import Client
    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    rtc_configuration = {
        "iceServers": token.ice_servers,
        "iceTransportPolicy": "relay",
    }
else:
    rtc_configuration = None

    
def response(audio: tuple[int, np.ndarray], conversation: list[dict], img: str | None):

    sampling_rate, audio_np = audio
    audio_np = audio_np.squeeze()

    audio_buffer = io.BytesIO()
    segment = AudioSegment(
        audio_np.tobytes(),
        frame_rate=sampling_rate,
        sample_width=audio_np.dtype.itemsize,
        channels=1,
    )

    segment.export(audio_buffer, format="wav")
    conversation.append({"role": "user", "content": gr.Audio((sampling_rate, audio_np))})
    conversation.append({"role": "assistant", "content": ""})

    base64_encoded = str(base64.b64encode(audio_buffer.getvalue()), encoding="utf-8")
    if API_URL is not None:
        output_audio_bytes = b""
        files = {"audio": base64_encoded}
        if img is not None:
            files["image"] = str(base64.b64encode(open(img, "rb").read()), encoding="utf-8")
        print("sending request to server")
        resp_text = ""
        with requests.post(API_URL, json=files, stream=True) as response:
            try:
                buffer = b''
                for chunk in response.iter_content(chunk_size=2048):
                    buffer += chunk
                    while b'\r\n--frame\r\n' in buffer:
                        frame, buffer = buffer.split(b'\r\n--frame\r\n', 1)
                        if b'Content-Type: audio/wav' in frame:
                            audio_data = frame.split(b'\r\n\r\n', 1)[1]
                            # audio_data = base64.b64decode(audio_data)
                            output_audio_bytes += audio_data
                            audio_array = np.frombuffer(audio_data, dtype=np.int16).reshape(1, -1)
                            yield (OUT_RATE, audio_array, "mono")
                        elif b'Content-Type: text/plain' in frame:
                            text_data = frame.split(b'\r\n\r\n', 1)[1].decode()
                            resp_text += text_data
                            if len(text_data) > 0:
                                conversation[-1]["content"] = resp_text
                                yield AdditionalOutputs(conversation)
            except Exception as e:
               raise Exception(f"Error during audio streaming: {e}") from e
            

def main(port=None):

    with gr.Blocks() as demo:
        gr.HTML(
            """
        <h1 style='text-align: center'>
        Mini-Omni-2 Chat (Powered by WebRTC ⚡️)
        </h1>
        """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        audio = WebRTC(
                            label="Stream",
                            rtc_configuration=rtc_configuration,
                            mode="send-receive",
                            modality="audio",
                        )
                    with gr.Column():
                        img = gr.Image(label="Image", type="filepath")
            with gr.Column():
                conversation = gr.Chatbot(label="Conversation", type="messages")
            
            audio.stream(
                fn=ReplyOnPause(
                    response, output_sample_rate=OUT_RATE, output_frame_size=480
                ),
                inputs=[audio, conversation, img],
                outputs=[audio],
                time_limit=90,
            )
            audio.on_additional_outputs(lambda c: c, outputs=[conversation])
    if port is not None:
        demo.queue().launch(share=False, server_name="0.0.0.0", server_port=port)
    else:
        demo.queue().launch()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
