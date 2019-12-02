import fire
import torch
import torch.nn as nn

from typing import Tuple
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_sound.models import build_model
from tacotron2_pytorch.dataset import get_datasets
from tacotron2_pytorch.trainer import Tacotron2Trainer


def main(meta_dir: str, save_dir: str,
         save_prefix: str, pretrained_path: str = '',
         model_name: str = 'tacotron2_base', batch_size: int = 32, num_workers: int = 16,
         lr: float = 1e-4, betas: Tuple[float] = (0.9, 0.99), weight_decay: float = 0.0,
         max_step: int = 100000, valid_max_step: int = 50, save_interval: int = 1000, log_interval: int = 50,
         grad_clip: float = 0.0, grad_norm: float = 30.0, gamma: float = 0.1, milestones: Tuple[int] = None,
         sr: int = 22050):

    # create model
    model = build_model(model_name).cuda()

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # create optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    if milestones:
        milestones = [int(x) for x in list(milestones)]
        scheduler = MultiStepLR(optimizer, milestones, gamma=gamma)
    else:
        scheduler = None

    train_loader, valid_loader = get_datasets(
        meta_dir, batch_size=batch_size, num_workers=num_workers
    )

    Tacotron2Trainer(
        model, optimizer, train_loader, valid_loader,
        max_step=max_step, valid_max_step=min(valid_max_step, len(valid_loader)), save_interval=save_interval,
        log_interval=log_interval,
        save_dir=save_dir, save_prefix=save_prefix, grad_clip=grad_clip, grad_norm=grad_norm,
        pretrained_path=pretrained_path, scheduler=scheduler, sr=sr
    ).run()


if __name__ == '__main__':
    fire.Fire(main)
