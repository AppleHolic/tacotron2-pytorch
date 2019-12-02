import fire
import numpy as np
import torch
import librosa
import os
from matplotlib.image import imsave
from pytorch_sound.utils.calculate import unnorm_mel
from pytorch_sound.utils.plots import imshow_to_buf, plot_to_buf
from pytorch_sound.utils.text import eng_t2i

from tacotron2_pytorch.utils import load_model, add_punctuation


def synthesize(text: str, pretrained_path: str, model_name: str, savedir: str):
    os.makedirs(savedir, exist_ok=True)

    # character indexing
    text = add_punctuation(text)
    chr_indices = np.array(eng_t2i(text))
    
    # to cuda tensor
    chr_tensor = torch.from_numpy(chr_indices).cuda()
    
    # load models
    model = load_model(model_name, pretrained_path)

    print('Load Vocoder ...')
    vocoder = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    vocoder = vocoder.remove_weightnorm(vocoder)
    vocoder = vocoder.cuda()
    vocoder.eval()
    
    # inference
    with torch.no_grad():
        raugh_spec, res_spec, _, attention_weights = model.inference(chr_tensor.unsqueeze(0))
        post_spec = raugh_spec + res_spec
        # un-normalization
        post_spec = unnorm_mel(post_spec)
        raugh_spec = unnorm_mel(raugh_spec)
        # vocoder
        pred_audio = vocoder.infer(post_spec)[0]
        pred_raugh_audio = vocoder.infer(raugh_spec)[0]

    # vocoding
    pred_wav = pred_audio.cpu().numpy()
    pred_raugh_wav = pred_raugh_audio.cpu().numpy()
    pred_mel = post_spec.cpu().numpy()[0]
    att = attention_weights.cpu().numpy()[0]

    # to plot
    mel_img = imshow_to_buf(pred_mel)
    att_img = imshow_to_buf(att)
    plot_img = plot_to_buf(pred_wav)

    # make file path
    audio_path = os.path.join(savedir, 'pre_audio.wav')
    audio_raugh_path = os.path.join(savedir, 'pred_raugh_audio.wav')
    mel_path = os.path.join(savedir, 'pre_mel.png')
    att_path = os.path.join(savedir, 'pre_att.png')
    audio_plot_path = os.path.join(savedir, 'pre_plot.png')

    imsave(mel_path, mel_img.transpose(1, 2, 0))
    imsave(att_path, att_img.transpose(1, 2, 0))
    imsave(audio_plot_path, plot_img.transpose(1, 2, 0))

    # save wav
    librosa.output.write_wav(audio_path, pred_wav.clip(-1., 1.), 22050)
    librosa.output.write_wav(audio_raugh_path, pred_raugh_wav.clip(-1, 1.), 22050)
        

if __name__ == '__main__':
    fire.Fire(synthesize)
