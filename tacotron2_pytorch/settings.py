from pytorch_sound.models import register_model_architecture
from pytorch_sound.data.eng_handler import symbols


@register_model_architecture('tacotron2', 'tacotron2_base')
def tacotron2_base():
    return {
        # encoder arguments
        'voc_size': len(symbols) + 1,
        'embedding_dim': 512,
        'encoder_layers': 3,
        'kernel_size': 5,
        'encoder_dim': 256,
        # decoder arguments
        'spec_dim': 80,
        'prenet_dim': 256,
        'decoder_dim': 1024,
        'attention_dim': 128,
        'location_feature_dim': 32,
        'location_kernel_size': 31,
        'postnet_layers': 3,
        'postnet_dim': 512,
        'postnet_kernel_size': 5
    }


@register_model_architecture('tacotron2', 'tacotron2_pre64')
def tacotron2_pre64():
    d = tacotron2_base()
    d.update({
        'prenet_dim': 64
    })
    return d
