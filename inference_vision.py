import os
import torch
from litgpt.generate.base import next_token_image_batch
import soundfile as sf
from utils.snac_utils import layershift, reconscruct_snac, reconstruct_tensors, get_time_str
from utils.snac_utils import get_snac, generate_audio_data
import clip
import inference
from tqdm import tqdm
from inference import OmniInference, load_model, load_audio, download_model
from inference import text_vocabsize, padded_text_vocabsize, get_text_stream
from PIL import Image


torch.set_printoptions(sci_mode=False)

_image = inference._image
_eoimage = inference._eoimage
_pad_t = inference._pad_t
_input_t = inference._input_t
_answer_t = inference._answer_t
_eot = inference._eot
_eoa = inference._eoa
_pad_a = inference._pad_a
_input_a = inference._input_a
_answer_a = inference._answer_a


def get_input_ids_ImageQA_ATBatch(mel, leng, whispermodel, device):
    
    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)
        audio_feature = whispermodel.embed_audio(mel)[0][:leng]
    
    audio_len = audio_feature.size(0)
    
    input_ids = []
    input_ids_item = [[] for i in range(8)]
    for i in range(7):
        input_ids_item[i] =  [layershift(_image,i)] + [layershift(_pad_a,i)] * 50 + [layershift(_eoimage,i)] 
        input_ids_item[i] += [layershift(_input_a,i)]+[layershift(_pad_a,i)]*(audio_len)+[layershift(_eoa,i)]
        input_ids_item[i] += [layershift(_answer_a,i)]

    input_ids_item[-1] = [_pad_t]* (52 + 2 + audio_len) + [_answer_t] 
    input_ids_item = [torch.tensor(item) for item in input_ids_item]

    input_ids.append(input_ids_item)

    input_ids_item = [[] for i in range(8)]
    for i in range(7):
        input_ids_item[i] =  [layershift(_image,i)] + [layershift(_pad_a,i)] * 50 + [layershift(_eoimage,i)] 
        input_ids_item[i] += [layershift(_input_a,i)]+[layershift(_pad_a,i)]*(audio_len)+[layershift(_eoa,i)] + [layershift(_pad_a,i)]

    input_ids_item[-1] = [_pad_t]* (52 + 2 + audio_len) + [_answer_t] 

    input_ids_item = [torch.tensor(item) for item in input_ids_item]
    input_ids.append(input_ids_item)

    stacked_inputids = [[] for _ in range(8)]
    for i in range(2):
        for j in range(8):
            stacked_inputids[j].append(input_ids[i][j])
    stacked_inputids = [torch.stack(tensors) for tensors in stacked_inputids]

    return torch.stack([audio_feature,audio_feature]), stacked_inputids

    
def load_clip_model(ckpt_dir, device):
    clip_model_path = ckpt_dir + "/ViT-B-32.pt"
    if not os.path.exists(clip_model_path):
        clip_model_path = "ViT-B/32"
    clipmodel, clippreprocess = clip.load(clip_model_path, device=device)
    return clipmodel, clippreprocess

    
class OmniVisionInference(OmniInference):

    def __init__(self, ckpt_dir='./checkpoint', device='cuda:0'):
        self.device = device
        if not os.path.exists(ckpt_dir):
            print(f"checkpoint directory {ckpt_dir} not found, downloading from huggingface")
            download_model(ckpt_dir)
        self.fabric, self.model, self.text_tokenizer, self.snacmodel, self.whispermodel = load_model(ckpt_dir, device)
        self.clipmodel, self.clippreprocess = load_clip_model(ckpt_dir, device)

    def warm_up(self, 
                audio_sample='./data/samples/vision_qa_audio.wav',
                image_sample='./data/samples/vision_qa_image.jpg'
        ):
        for _ in self.run_vision_AA_batch_stream(audio_sample, image_sample, 
                                                 save_path="./data/samples/vision_qa_output.wav",
                                                 warm_up=True):
            pass

    @torch.inference_mode()
    def run_vision_AA_batch_stream(self, audio_path, image_path, 
                                stream_stride=4,
                                max_returned_tokens=2048, 
                                temperature=0.9, 
                                top_k=1, 
                                top_p=1.0,
                                eos_id_a=_eoa, 
                                eos_id_t=_eot, 
                                pad_id=_pad_t,
                                save_path=None,
                                warm_up=False
        ):
        with self.fabric.init_tensor():
            self.model.set_kv_cache(batch_size=2)

        model = self.model

        mel, leng = load_audio(audio_path)
        img = Image.open(image_path)

        audio_feature, input_ids = get_input_ids_ImageQA_ATBatch(mel, leng, self.whispermodel, self.device)
        ima = self.clippreprocess(img).unsqueeze(0).to(self.device)
        ima_feature = self.clipmodel.encode_image(ima).squeeze(0).to(self.device)
        
        ima_feature = torch.stack([ima_feature.clone(),ima_feature.clone()]).to(self.device)
        leng = [leng,leng]
        task = ['ImageQA_A','ImageQA_AT']

        T = input_ids[0].size(1)  
        assert max_returned_tokens > T, f"max_returned_tokens {max_returned_tokens} should be greater than audio length {T}"

        if model.max_seq_length < max_returned_tokens - 1:
            raise NotImplementedError(
                f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
            )

        list_output = [[] for i in range(8)]

        tokens_A , token_T = next_token_image_batch(
            model, 
            audio_feature.to(torch.float32).to(self.device),
            ima_feature.to(torch.float32).to(self.device) , 
            input_ids , 
            whisper_lens = leng , 
            task = task, 
            input_pos = torch.arange(0, T, device=self.device), 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p
        )
        for i in range(7): list_output[i].append(tokens_A[i].tolist()[0])
        list_output[7].append(token_T.tolist()[0])

        text_end = False
        index = 1
        nums_generate = stream_stride
        begin_generate = False
        current_index = 0
        input_pos = torch.tensor([T], device=self.device)

        model_input_ids = [[] for i in range(8)]
        for i in range(7):
            tokens_A[i] = tokens_A[i].clone() + padded_text_vocabsize+ i * 4160
            model_input_ids[i].append(tokens_A[i].clone().to(self.device).to(torch.int32))
            model_input_ids[i].append(torch.tensor([layershift(4097,i)],device=self.device))
            model_input_ids[i] = torch.stack(model_input_ids[i])
        
        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1] = torch.stack(model_input_ids[-1])

        text_index = 0
        is_text_end = False

        for _ in tqdm(range(2, max_returned_tokens - T + 1)):
            
            tokens_A , token_T = next_token_image_batch(model, None , None , 
                                                        input_ids = model_input_ids, 
                                                        whisper_lens= None, 
                                                        task = None, 
                                                        input_pos = input_pos, 
                                                        temperature=temperature, 
                                                        top_k=top_k, 
                                                        top_p=top_p)

            if text_end:
                token_T = torch.tensor([_pad_t], device=self.device)

            if tokens_A[-1] == eos_id_a:
                break
            if token_T == eos_id_t:
                text_end = True

            for i in range(7): list_output[i].append(tokens_A[i].tolist()[0])
            list_output[7].append(token_T.tolist()[0])
            

            if index == 7:
                begin_generate = True
            
            if begin_generate:
                current_index += 1
                if current_index == nums_generate:
                    current_index = 0
                    snac = get_snac(list_output,index,nums_generate)
                    audio_stream = generate_audio_data(snac, self.snacmodel, self.device)
                    if is_text_end:
                        text_stream = ""
                    else:
                        text_stream, text_index, is_text_end = get_text_stream(list_output, text_index, self.text_tokenizer)

                    yield (audio_stream, text_stream)

                    if warm_up:
                        break

            input_pos = input_pos.add_(1)
            model_input_ids = [[] for i in range(8)]
            for i in range(7):
                tokens_A[i] = tokens_A[i].clone() + padded_text_vocabsize+ i * 4160
                model_input_ids[i].append(tokens_A[i].clone().to(self.device).to(torch.int32))
                model_input_ids[i].append(torch.tensor([layershift(4097,i)],device=self.device))
                model_input_ids[i] = torch.stack(model_input_ids[i])
            
            model_input_ids[-1].append(token_T.clone().to(torch.int32))
            model_input_ids[-1].append(token_T.clone().to(torch.int32))
            model_input_ids[-1] = torch.stack(model_input_ids[-1])

            index += 1    
            
        text_tokens = list_output[-1]
        if text_vocabsize in text_tokens:
            text_tokens = text_tokens[:text_tokens.index(text_vocabsize)]
        res_text = self.text_tokenizer.decode(torch.tensor(text_tokens))
        print(f"text output: {res_text}")

        if save_path is not None:
            audiolist = reconscruct_snac(list_output)
            audio = reconstruct_tensors(audiolist)
            with torch.inference_mode():
                audio_hat = self.snacmodel.decode(audio)
                sf.write(save_path, audio_hat.squeeze().cpu().numpy(), 24000)

        model.clear_kv_cache()

        
def test_vision_infer():
    client = OmniVisionInference()
    client.warm_up()
    input_audio_path = './data/samples/vision_qa_audio.wav'
    input_image_path = './data/samples/vision_qa_image.jpg'

    res_text = ""
    for audio_stream, text_stream in client.run_vision_AA_batch_stream(
        input_audio_path, 
        input_image_path,
        save_path="./vision_qa_output.wav"
    ):
        res_text += text_stream
    print(f"text_output: {res_text}")


if __name__ == "__main__":
    test_vision_infer()
