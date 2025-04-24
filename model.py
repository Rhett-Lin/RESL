import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
import numpy as np
import clip
import torch
from torch import nn
from timm.models.layers import trunc_normal_


class CLIPModel(nn.Module):
    def __init__(self, model_name="ViT-B/32"):
        super().__init__()
        self.clip, self.preprocess = clip.load(model_name, device="cuda")

    @property
    def dtype(self):
        return self.clip.visual.conv1.weight.dtype

    def encode_image(self, image):
        image_features = self.clip.visual(image.type(self.dtype))
        return image_features

    def encode_text(self, text):
        x = self.clip.token_embedding(text).type(self.dtype)
        x = x + self.clip.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection
        return x


class Encoder(nn.Module):
    def __init__(self, in_channel):
        super(Encoder, self).__init__()
        self.in_channel = in_channel
        self.net = nn.Sequential(
            nn.Linear(self.in_channel, self.in_channel),
            nn.BatchNorm1d(self.in_channel),
            nn.ReLU(),
            nn.Linear(self.in_channel, self.in_channel),
            nn.BatchNorm1d(self.in_channel),
            nn.ReLU()
        )
        self.double_line = nn.Linear(self.in_channel, self.in_channel * 2)

    def forward(self, *input):
        x = self.net(*input)
        params = self.double_line(x)
        mu, sigma = params[:, :int(self.in_channel)], params[:, int(self.in_channel):]
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)


class Model(Encoder):
    def __init__(self, num_heads, output_dims, in_channel=512):
        super(Model, self).__init__(in_channel)
        self.in_channel = in_channel
        self.num_heads = num_heads
        self.output_dims = output_dims

        assert len(output_dims) == num_heads

        self.cluster_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.in_channel, output_dim)
            ) for output_dim in output_dims
        ])

        self.encoder = Encoder(in_channel)
        _initialize_weights(self)

    def forward(self, input, forward_pass='output_i'):
        if forward_pass == 'output_i':
            x = self.encoder.net(input)
            
            fea = [head(x) for head in self.cluster_heads]
            outputs = [torch.softmax(head(x), dim=1) for head in self.cluster_heads]

            return outputs, self.encoder

        elif forward_pass == 'head_i':
            outputs = [torch.softmax(head(input), dim=1) for head in self.cluster_heads]
            return outputs


def _initialize_weights(self):
    print("initialize")
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            assert hasattr(self, 'batchnorm_track')
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def UD_constraint(model, data, num_heads):
    outputs, _ = model(data)
    classer = outputs[num_heads-1]
    CL = classer.detach().cpu().numpy()
    N, K = CL.shape
    CL = CL.T

    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    CL **= 10
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e3
    _counter = 0

    while err > 1e-2 and _counter < 75:
        r = inv_K / (CL @ c)
        c_new = inv_N / (r.T @ CL).T
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1

    CL *= np.squeeze(c)
    CL = CL.T
    CL *= np.squeeze(r)
    CL = CL.T

    argmaxes = np.nanargmax(CL, 0)
    newL = torch.LongTensor(argmaxes)
    return newL
