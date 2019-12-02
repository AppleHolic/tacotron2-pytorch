import torch
from pytorch_sound.models import build_model
from pytorch_sound.utils.commons import get_loadable_checkpoint


def load_model(model_name: str, pretrained_path: str) -> torch.nn.Module:
    print('Load model ...')
    model = build_model(model_name).cuda()
    chk = torch.load(pretrained_path)['model']
    model.load_state_dict(get_loadable_checkpoint(chk))
    model.eval()
    return model


def add_punctuation(sentence: str):
    if sentence[-1] not in '.!?':
        return sentence + '.'
    else:
        return sentence
