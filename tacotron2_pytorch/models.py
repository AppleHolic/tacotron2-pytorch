import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_sound.models import register_model


class Encoder(nn.Module):

    def __init__(self, voc_size: int, embedding_dim: int, encoder_layers: int, kernel_size: int, encoder_dim: int):
        super().__init__()
        # embedding
        self.embedding = nn.Embedding(voc_size, embedding_dim, padding_idx=0)

        # conv
        self.conv = nn.Sequential(*[
            nn.Sequential(
                *[nn.Conv1d(embedding_dim if not idx else encoder_dim, encoder_dim, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(encoder_dim), nn.ReLU()]
            ) for idx in range(encoder_layers)
        ])

        # bi-directional lstm
        self.lstm = nn.LSTM(encoder_dim, encoder_dim // 2, 1, bidirectional=True, batch_first=True)

    def forward(self, chr_tensor: torch.Tensor):
        # embedding
        x = self.embedding(chr_tensor)
        x = x.transpose(1, 2)

        # conv
        x = self.conv(x)

        # lstm
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x


class PostNet(nn.Module):

    def __init__(self, spec_dim: int, postnet_layers: int, postnet_dim: int, postnet_kernel_size: int):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(spec_dim, postnet_dim, 1),
            nn.BatchNorm1d(postnet_dim),
            nn.Tanh()
        )
        self.body = nn.Sequential(*[
            nn.Conv1d(postnet_dim, postnet_dim, postnet_kernel_size, padding=postnet_kernel_size // 2),
            nn.BatchNorm1d(postnet_dim),
            nn.Tanh()
        ] * postnet_layers)
        self.out = nn.Conv1d(postnet_dim, spec_dim, 1)

    def forward(self, raugh_spec: torch.Tensor):
        x = self.in_conv(raugh_spec)
        x = self.body(x)
        return self.out(x)


class StaticDropout(nn.Module):

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, self.p, training=True)


class Decoder(nn.Module):
    """Decoder is an autoregressive recurrent neural network which predicts a
    (mel) spectrogram from the encoded input sequence one frame at a time.
    """

    def __init__(self, spec_dim: int, encoder_dim: int, prenet_dim: int,
                 decoder_dim: int, attention_dim: int, location_feature_dim: int, location_kernel_size: int,
                 postnet_layers: int, postnet_dim: int, postnet_kernel_size: int):
        super(Decoder, self).__init__()
        self.spec_dim = spec_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        # pre net
        self.pre_net = nn.Sequential(
            nn.Conv1d(spec_dim, prenet_dim, 1),
            StaticDropout(0.5),
            nn.ReLU(),
            nn.Conv1d(prenet_dim, prenet_dim, 1),
            StaticDropout(0.5),
            nn.ReLU(),
        )

        # location sensitive attention
        self.attention = LocationSensitiveAttention(attention_dim,
                                                    decoder_dim,
                                                    encoder_dim,
                                                    location_feature_dim,
                                                    location_kernel_size)

        self.rnn = nn.ModuleList([
            nn.LSTMCell(prenet_dim + encoder_dim, decoder_dim),
            nn.LSTMCell(decoder_dim, decoder_dim)
        ])

        self.feature_linear = nn.Sequential(
            nn.Linear(decoder_dim + encoder_dim, spec_dim),
            nn.Tanh()
        )

        # stop
        self.stop_linear = nn.Linear(decoder_dim + encoder_dim, 1)

        # post net
        self.post_net = PostNet(spec_dim, postnet_layers, postnet_dim, postnet_kernel_size)

        # states
        self.h_list = None
        self.c_list = None
        self.cumulative_attention_weight = None
        self.attention_context = None

    def initialize(self, tensor: torch.Tensor):
        size = tensor.size()
        batch_size, Ti = size[0], size[2]
        # Init LSTMCell state
        self.h_list = [torch.zeros(batch_size, self.decoder_dim).type_as(tensor)] * 2
        self.c_list = [torch.zeros(batch_size, self.decoder_dim).type_as(tensor)] * 2
        # Init attention
        self.cumulative_attention_weight = torch.zeros(batch_size, Ti).type_as(tensor)
        self.attention_context = torch.zeros(batch_size, self.encoder_dim).type_as(tensor)
        self.attention.reset()

    def forward(self, encoder_feature: torch.Tensor, encoder_mask: torch.Tensor, spectrogram: torch.Tensor,
                decoder_mask: torch.Tensor):
        # init rnn state and attention
        self.initialize(encoder_feature)

        # Forward
        prenet_out = self.pre_net(spectrogram)

        feat_outputs, stop_tokens, attention_weights = [], [], []
        time_len = spectrogram.size(2)

        for t in range(time_len):
            step_input = prenet_out[..., t]
            feat_output, stop_token, attention_weight = self.step(step_input, encoder_feature, encoder_mask)

            feat_outputs += [feat_output]
            stop_tokens += [stop_token]
            attention_weights += [attention_weight]

        feat_outputs = torch.stack(feat_outputs, dim=2)
        stop_tokens = torch.cat(stop_tokens, dim=1)
        attention_weights = torch.stack(attention_weights, dim=2)

        # post net
        feat_residual_outputs = self.post_net(feat_outputs)

        # Mask
        decoder_mask = decoder_mask.unsqueeze(1)  # [N, To, 1]
        feat_outputs = feat_outputs.masked_fill(decoder_mask, -1.)
        feat_residual_outputs = feat_residual_outputs.masked_fill(decoder_mask, 0.)
        attention_weights = attention_weights.masked_fill(decoder_mask, 0.)
        stop_tokens = stop_tokens.masked_fill(decoder_mask.squeeze(), 1e3)  # sigmoid(1e3) = 1, log(1) = 0

        return feat_outputs, feat_residual_outputs, stop_tokens, attention_weights

    def inference(self, encoder_feature: torch.Tensor):
        """Inference one utterance."""
        # max len by character length
        time_len = encoder_feature.size(2)
        max_decoder_steps = time_len * (2 ** 4)

        # Init
        # get go frame
        go_frame = torch.ones(encoder_feature.size(0), self.spec_dim, 1).type_as(encoder_feature) * (-1)

        # init rnn state and attention
        self.initialize(encoder_feature)

        # Forward
        feat_outputs, stop_tokens, attention_weights = [], [], []
        step_input = go_frame
        self.attention.reset()

        # loop until stop token
        while True:
            step_input = self.pre_net(step_input).squeeze(2)
            feat_output, stop_token, attention_weight = self.step(step_input, encoder_feature, None)
            # record
            feat_outputs += [feat_output]
            stop_tokens += [stop_token]
            attention_weights += [attention_weight]

            # terminate?
            stop = torch.sigmoid(stop_token).item()
            if stop > 0.5:
                break
            elif torch.argmax(attention_weight).item() == time_len - 2:
                break
            elif len(feat_outputs) == max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            # autoregressive
            step_input = feat_output.unsqueeze(2)

        feat_outputs = torch.stack(feat_outputs, dim=2)
        stop_tokens = torch.stack(stop_tokens, dim=2).squeeze()
        attention_weights = torch.stack(attention_weights, dim=2)
        feat_residual_outputs = self.post_net(feat_outputs)

        return feat_outputs, feat_residual_outputs, stop_tokens, attention_weights

    def step(self, step_input: torch.tensor, encoder_feature: torch.Tensor, encoder_mask: torch.Tensor):
        rnn_input = torch.cat((step_input, self.attention_context), dim=1)
        self.h_list[0], self.c_list[0] = self.rnn[0](rnn_input, (self.h_list[0], self.c_list[0]))
        self.h_list[1], self.c_list[1] = self.rnn[1](self.h_list[0], (self.h_list[1], self.c_list[1]))
        rnn_output = self.h_list[1]

        self.attention_context, attention_weight = self.attention(rnn_output,
                                                                  encoder_feature,
                                                                  self.cumulative_attention_weight,
                                                                  mask=encoder_mask)
        self.cumulative_attention_weight = self.cumulative_attention_weight + attention_weight

        linear_input = torch.cat((rnn_output, self.attention_context), dim=1)
        feat_output = self.feature_linear(linear_input)
        stop_token = self.stop_linear(linear_input)
        return feat_output, stop_token, attention_weight


class LocationSensitiveAttention(nn.Module):

    def __init__(self, attention_dim: int, decoder_dim: int,
                 encoder_dim: int, location_dim: int, location_kernel_size: int):
        super().__init__()
        self.W = nn.Conv1d(decoder_dim, attention_dim, 1, bias=True)  # keep one bias
        self.V = nn.Conv1d(encoder_dim, attention_dim, 1, bias=False)
        self.U = nn.Conv1d(location_dim, attention_dim, 1, bias=False)
        self.F = nn.Conv1d(in_channels=1, out_channels=location_dim,
                           kernel_size=location_kernel_size, stride=1, padding=location_kernel_size // 2,
                           bias=False)
        self.v = nn.Conv1d(attention_dim, 1, 1, bias=False)

        self.Vh = None

    def reset(self):
        """Remember to reset at decoder step 0"""
        # pre-compute V*h_j due to it is independent from the decoding step i
        self.Vh = None

    def get_energy(
            self, query: torch.Tensor, values: torch.Tensor, cumulative_attention_weights: torch.Tensor, mask=None):
        query = query.unsqueeze(2)
        Ws = self.W(query)
        if self.Vh is None:
            self.Vh = self.V(values)
        location_feature = self.F(cumulative_attention_weights.unsqueeze(1))  # [N, 32, Ti]
        Uf = self.U(location_feature)
        energies = self.v(torch.tanh(Ws + self.Vh + Uf)).squeeze(1)  # [N, Ti]
        if mask is not None:
            energies = energies.masked_fill(mask, -np.inf)
        return energies

    def forward(self, query: torch.Tensor, values: torch.Tensor,
                cumulative_attention_weights: torch.Tensor, mask: torch.Tensor = None):
        energies = self.get_energy(query, values, cumulative_attention_weights, mask)  # [N, Ti]
        attention_weights = F.softmax(energies, dim=1)  # [N, Ti]
        attention_context = torch.bmm(values, attention_weights.unsqueeze(2))
        attention_context = attention_context.squeeze(2)  # [N, Ti]
        return attention_context, attention_weights


@register_model('tacotron2')
class Tacotron2(nn.Module):

    def __init__(
            self, voc_size: int, embedding_dim: int, encoder_layers: int, kernel_size: int, encoder_dim: int,
            spec_dim: int, prenet_dim: int, decoder_dim: int, attention_dim: int, location_feature_dim: int,
            location_kernel_size: int, postnet_layers: int, postnet_dim: int, postnet_kernel_size: int
    ):
        super().__init__()
        self.encoder = Encoder(voc_size, embedding_dim, encoder_layers, kernel_size, encoder_dim)
        self.decoder = Decoder(spec_dim, encoder_dim, prenet_dim, decoder_dim, attention_dim, location_feature_dim,
                               location_kernel_size, postnet_layers, postnet_dim, postnet_kernel_size)

    def forward(self, chr_tensor: torch.Tensor, spectrogram: torch.Tensor, chr_mask: torch.Tensor, spec_mask: torch.Tensor):
        # encoder
        encoder_feature = self.encoder(chr_tensor)
        # decoder
        raugh_spec, res_spec, stop_tokens, attention_weights \
            = self.decoder(encoder_feature, chr_mask, spectrogram, spec_mask)
        return raugh_spec, res_spec, stop_tokens, attention_weights

    def inference(self, chr_tensor: torch.Tensor):
        encoder_padded_outputs = self.encoder(chr_tensor)
        raugh_spec, res_spec, stop_tokens, attention_weights \
            = self.decoder.inference(encoder_padded_outputs)
        return raugh_spec, res_spec, stop_tokens, attention_weights
