import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vad import get_speech_timestamps, collect_chunks, VadOptions

import streamlit as st
import wave

# from ASR import recognize
import requests
import pyaudio
import numpy as np
import base64
import io
from typing import List
import av
import os
import time
import tempfile
import librosa
import traceback
from pydub import AudioSegment
from datetime import datetime
from PIL import Image
import streamlit_webrtc
from io import BytesIO

# set wide mode
# st.set_page_config(layout="wide")


last_video_frame = None
last_video_frame_ts = time.time()


API_URL = os.getenv("API_URL", "http://127.0.0.1:60808/chat")
API_URL = None if API_URL == "" else API_URL

# recording parameters
IN_FORMAT = pyaudio.paInt16
IN_CHANNELS = 1
IN_RATE = 24000
IN_CHUNK = 1024
IN_SAMPLE_WIDTH = 2
VAD_STRIDE = 0.5

# playing parameters
OUT_FORMAT = pyaudio.paInt16
OUT_CHANNELS = 1
OUT_RATE = 24000
OUT_SAMPLE_WIDTH = 2
OUT_CHUNK = 5760


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def run_vad(ori_audio, sr):
    _st = time.time()
    try:
        audio = np.frombuffer(ori_audio, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        sampling_rate = 16000
        if sr != sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)

        vad_parameters = {}
        vad_parameters = VadOptions(**vad_parameters)
        speech_chunks = get_speech_timestamps(audio, vad_parameters)
        audio = collect_chunks(audio, speech_chunks)
        duration_after_vad = audio.shape[0] / sampling_rate

        if sr != sampling_rate:
            # resample to original sampling rate
            vad_audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=sr)
        else:
            vad_audio = audio
        vad_audio = np.round(vad_audio * 32768.0).astype(np.int16)
        vad_audio_bytes = vad_audio.tobytes()

        return duration_after_vad, vad_audio_bytes, round(time.time() - _st, 4)
    except Exception as e:
        msg = f"[asr vad error] audio_len: {len(ori_audio)/(sr*2):.3f} s, trace: {traceback.format_exc()}"
        print(msg)
        return -1, ori_audio, round(time.time() - _st, 4)


def warm_up():
    frames = b"\x00\x00" * 1024 * 2  # 1024 frames of 2 bytes each
    dur, frames, tcost = run_vad(frames, 16000)
    print(f"warm up done, time_cost: {tcost:.3f} s")


def save_tmp_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        file_name = tmpfile.name
        audio = AudioSegment(
            data=audio_bytes,
            sample_width=OUT_SAMPLE_WIDTH,
            frame_rate=OUT_RATE,
            channels=OUT_CHANNELS,
        )
        audio.export(file_name, format="wav")
        return file_name


def speaking(status, resp_text_holder=None, encoded_img=None):

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open PyAudio stream
    stream = p.open(
        format=OUT_FORMAT, channels=OUT_CHANNELS, rate=OUT_RATE, output=True
    )

    audio_buffer = io.BytesIO()
    wf = wave.open(audio_buffer, "wb")
    wf.setnchannels(IN_CHANNELS)
    wf.setsampwidth(IN_SAMPLE_WIDTH)
    wf.setframerate(IN_RATE)
    total_frames = b"".join(st.session_state.frames)
    dur = len(total_frames) / (IN_RATE * IN_CHANNELS * IN_SAMPLE_WIDTH)
    status.warning(f"Speaking... recorded audio duration: {dur:.3f} s")
    wf.writeframes(total_frames)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        with open(tmpfile.name, "wb") as f:
            f.write(audio_buffer.getvalue())

        with open("input_audio.wav", "wb") as f:
            f.write(audio_buffer.getvalue())

        file_name = tmpfile.name
        with st.chat_message("user"):
            st.audio(file_name, format="audio/wav", loop=False, autoplay=False)
        st.session_state.messages.append(
            {"role": "assistant", "content": file_name, "type": "audio"}
        )

    st.session_state.frames = []

    audio_bytes = audio_buffer.getvalue()
    base64_encoded = str(base64.b64encode(audio_bytes), encoding="utf-8")
    if API_URL is not None:
        output_audio_bytes = b""
        files = {"audio": base64_encoded}
        if encoded_img is not None:
            files["image"] = encoded_img
        print("sending request to server")
        resp_text_holder.empty()
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
                            audio_array = np.frombuffer(audio_data, dtype=np.int8)
                            stream.write(audio_array)
                        elif b'Content-Type: text/plain' in frame:
                            text_data = frame.split(b'\r\n\r\n', 1)[1].decode()
                            resp_text += text_data
                            if len(text_data) > 0:
                                print(resp_text, end='\r')
                                resp_text_holder.write(resp_text)

            except Exception as e:
                st.error(f"Error during audio streaming: {e}")

        out_file = save_tmp_audio(output_audio_bytes)
        with st.chat_message("assistant"):
            st.write(resp_text)
        with st.chat_message("assistant"):
            st.audio(out_file, format="audio/wav", loop=False, autoplay=False)
        st.session_state.messages.append(
            {"role": "assistant", "content": resp_text, "type": "text"}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": out_file, "type": "audio"}
        )
    else:
        st.error("API_URL is not set. Please set the API_URL environment variable.")
        time.sleep(1)

    wf.close()
    # Close PyAudio stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
    st.session_state.speaking = False
    st.session_state.recording = True


def recording(status):
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=IN_FORMAT,
        channels=IN_CHANNELS,
        rate=IN_RATE,
        input=True,
        frames_per_buffer=IN_CHUNK,
    )

    temp_audio = b""
    vad_audio = b""

    start_talking = False
    last_temp_audio = None
    st.session_state.frames = []

    while st.session_state.recording:
        status.success("Listening...")
        audio_bytes = stream.read(IN_CHUNK)
        temp_audio += audio_bytes

        if len(temp_audio) > IN_SAMPLE_WIDTH * IN_RATE * IN_CHANNELS * VAD_STRIDE:
            dur_vad, vad_audio_bytes, time_vad = run_vad(temp_audio, IN_RATE)

            print(f"duration_after_vad: {dur_vad:.3f} s, time_vad: {time_vad:.3f} s")

            if dur_vad > 0.2 and not start_talking:
                if last_temp_audio is not None:
                    st.session_state.frames.append(last_temp_audio)
                start_talking = True
            if start_talking:
                st.session_state.frames.append(temp_audio)
            if dur_vad < 0.1 and start_talking:
                st.session_state.recording = False
                print(f"speech end detected. excit")
            last_temp_audio = temp_audio
            temp_audio = b""

    stream.stop_stream()
    stream.close()

    audio.terminate()


async def queued_video_frames_callback(frames: List[av.VideoFrame]) -> List[av.VideoFrame]:
    # print(f"test-------queued_video_frames_callback")
    global last_video_frame
    global last_video_frame_ts
    if len(frames) != 0:
        if time.time() - last_video_frame_ts > 1:
            last_frame = frames[-1]
            # with video_frame_lock:
            #     last_video_frame[0] = last_frame.to_image()
            #     last_video_frame_ts[0] = time.time()
            last_video_frame = last_frame.to_image()
            last_video_frame_ts = time.time()

    return frames



def main():

    st.title("Chat Mini-Omni2 Demo")
    status = st.empty()

    # Mode selection
    mode = st.radio(
        "Select mode:",
        ("Audio-only", "Audio-vision"),
        key="mode_selection",
        horizontal=True
    )

    if mode == "Audio-only":
        st.session_state.use_vision = False
        st.info("Audio-only mode selected. The system will process only audio input.")
    else:  # Audio-vision
        st.session_state.use_vision = True
        st.info("Audio-vision mode selected. The system will process both audio and video input.")

    if "warm_up" not in st.session_state:
        warm_up()
        st.session_state.warm_up = True
    if "start" not in st.session_state:
        st.session_state.start = False
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "speaking" not in st.session_state:
        st.session_state.speaking = False
    if "frames" not in st.session_state:
        st.session_state.frames = []

    if not st.session_state.start:
        status.warning("Click Start to chat")

    start_col, stop_col, _ = st.columns([0.2, 0.2, 0.6])
    start_button = start_col.button("Start", key="start_button")
    # stop_button = stop_col.button("Stop", key="stop_button")
    if start_button:
        time.sleep(1)
        st.session_state.recording = True
        st.session_state.start = True

    if st.session_state.use_vision:
        with st.sidebar:

            webrtc_ctx = streamlit_webrtc.webrtc_streamer(
                    key="speech-w-video",
                    mode=streamlit_webrtc.WebRtcMode.SENDRECV,
                    # rtc_configuration={"iceServers": get_ice_servers()},
                    media_stream_constraints={"video": True, "audio": False},
                    # video_receiver_size=10,  # Increased from default 4 to 10
                    queued_video_frames_callback=queued_video_frames_callback,
                )

            if not webrtc_ctx.state.playing:
                st.warning("Please allow camera access and try again.")
                return

    resp_text_holder = st.empty()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"])
            elif message["type"] == "img":
                st.image(message["content"], width=300)
            elif message["type"] == "audio":
                st.audio(
                    message["content"], format="audio/wav", loop=False, autoplay=False
                )

    while st.session_state.start:
        if st.session_state.recording:
            recording(status)

        if not st.session_state.recording and st.session_state.start:
            encoded_img = None
            if st.session_state.use_vision:
                # last_img = webrtc_ctx.video_receiver.get_frame(timeout=5).to_image()
                last_img = last_video_frame
                if last_img:
                    with st.chat_message("user"):
                        st.image(last_img, width=300)
                    st.session_state.messages.append({"role": "user", "content": last_img, "type": "img"})

                    input_img = last_img
                    buffer = BytesIO()
                    input_img.save(buffer, format="JPEG")

                    with open("input_image.jpg", "wb") as f:
                        f.write(buffer.getvalue())

                    encoded_img = str(base64.b64encode(buffer.getvalue()), encoding="utf-8")
                else:
                    st.error("No image captured. Please allow camera access and try again.")
                    return

            st.session_state.speaking = True
            speaking(status, resp_text_holder, encoded_img)

        # if stop_button:
        #     status.warning("Stopped, click Start to chat")
        #     st.session_state.start = False
        #     st.session_state.recording = False
        #     st.session_state.frames = []
        #     break


if __name__ == "__main__":
    main()
