import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_sound import settings
from pytorch_sound.models.transforms import LogMelSpectrogram, MelMasker

from pytorch_sound.trainer import Trainer, LogType
from pytorch_sound.utils.calculate import norm_mel, unnorm_mel


class Tacotron2Trainer(Trainer):

    def __init__(self, model: nn.Module, optimizer,
                 train_dataset, valid_dataset,
                 max_step: int, valid_max_step: int, save_interval: int, log_interval: int,
                 save_dir: str, save_prefix: str = '',
                 grad_clip: float = 0.0, grad_norm: float = 0.0,
                 sr: int = 22050,
                 pretrained_path: str = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None):
        super().__init__(model, optimizer, train_dataset, valid_dataset,
                         max_step, valid_max_step, save_interval, log_interval, save_dir, save_prefix,
                         grad_clip, grad_norm, pretrained_path, sr=sr, scheduler=scheduler)
        # log mel spec
        self.mel_func = LogMelSpectrogram(
            sr, settings.MEL_SIZE, settings.WIN_LENGTH, settings.WIN_LENGTH, settings.HOP_LENGTH,
            float(settings.MIN_DB), float(settings.MAX_DB), float(settings.MEL_MIN), float(settings.MEL_MAX)
        ).cuda()

        # loss
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        # vocoder
        self.vocoder = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
        self.vocoder = self.vocoder.remove_weightnorm(self.vocoder)
        self.vocoder = self.vocoder.cuda()
        self.vocoder.eval()

        # mel masker
        self.mel_masker = MelMasker(settings.WIN_LENGTH, settings.HOP_LENGTH)

    def calc_mel_loss(self, pred_mel: torch.Tensor, target_mel: torch.Tensor):
        # padding to match time dimension
        origin_size, buf_size = target_mel.size(2), pred_mel.size(2)
        # assert origin_size < buf_size, f'Origin Size : {origin_size} must be smaller than buffer size {buf_size}!'
        if origin_size >= buf_size:
            pad_mel = target_mel[..., :buf_size]
        else:
            pad_mel = F.pad(target_mel, (0, buf_size - origin_size), mode='constant', value=-1)

        loss = self.mse_loss(pad_mel, pred_mel)
        return loss

    def slice_or_not(self, pred_mel: torch.Tensor, mel: torch.Tensor):
        # mel slicing
        pred_len = pred_mel.size(-1)
        target_len = mel.size(-1)
        if pred_len < target_len:
            mel = mel[..., :pred_len]
        else:
            pred_mel = pred_mel[..., :target_len]
        return pred_mel, mel

    def forward(self, wav, chr_indices, wav_mask, chr_mask, is_logging: bool = False):
        # wav to mel
        mel = self.mel_func(wav)
        mel = norm_mel(mel)

        with torch.no_grad():
            mel_mask = torch.abs(self.mel_masker(wav) - 1)

        # mel shift
        p = torch.ones(*mel.size()[:2], 1).type_as(mel) * -1
        shifted_mel = torch.cat([p, mel[..., :-1]], dim=2)

        result = self.model(chr_indices, shifted_mel, (chr_mask - 1).bool(), mel_mask.bool())
        raugh_spec, res_spec, stop_tokens, attention_weights = result

        # losses
        stop_loss = self.bce_loss(torch.sigmoid(stop_tokens), mel_mask)
        post_spec = raugh_spec + res_spec

        # loss
        raugh_loss = self.calc_mel_loss(raugh_spec, mel)
        post_loss = self.calc_mel_loss(post_spec, mel)
        loss = raugh_loss + post_loss + stop_loss

        if is_logging:
            target_audio = wav[0]
            with torch.no_grad():
                post_mel = unnorm_mel(post_spec[:1])
                pred_audio = self.vocoder.infer(post_mel)[0]

            target_mel = mel[0]
            post_mel = post_mel[0]
            raugh_mel = raugh_spec[0]
            att = attention_weights[0]

            meta = {
                'loss': (loss.item(), LogType.SCALAR),
                'raugh_loss': (raugh_loss.item(), LogType.SCALAR),
                'post_loss': (post_loss.item(), LogType.SCALAR),
                'stop_loss': (stop_loss.item(), LogType.SCALAR),
                'target_mel': (target_mel, LogType.IMAGE),
                'raugh_mel': (raugh_mel, LogType.IMAGE),
                'post_mel': (post_mel, LogType.IMAGE),
                'attention': (att, LogType.IMAGE),
                'target_audio.audio': (target_audio, LogType.AUDIO),
                'pred_audio.audio': (pred_audio, LogType.AUDIO),
                'target_audio.plot': (target_audio, LogType.PLOT),
                'pred_audio.plot': (pred_audio, LogType.PLOT),
            }
        else:
            meta = {}

        return loss, meta
