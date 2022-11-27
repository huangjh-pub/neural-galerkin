import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torch_spsr.ext import sparse_op


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                        freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, d_in=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': d_in,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ConvolutionFunction(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        nbmaps: torch.Tensor,
        nbsizes: torch.Tensor,
        sizes: Tuple[int, int],
        transposed: bool = False,
        no_acc: bool = False
    ) -> torch.Tensor:
        input = input.contiguous()
        weight = weight.contiguous()
        nbmaps = nbmaps.int().contiguous()
        nbsizes = nbsizes.int().contiguous()

        if not transposed:
            output = torch.zeros(
                sizes[1],
                weight.size(-1),
                dtype=input.dtype,
                device=input.device,
            )
        else:
            output = torch.zeros(
                sizes[0],
                weight.size(-1),
                dtype=input.dtype,
                device=input.device,
            )

        if input.device.type == 'cuda':
            sparse_op.convolution_forward_cuda(
                input,
                output,
                weight,
                nbmaps,
                nbsizes.cpu(),
                transposed,
                no_acc
            )
        else:
            a = 0
            for k in range(weight.shape[0]):
                b = a + nbsizes[k]
                if not transposed:
                    i = nbmaps[a:b, 0].long()
                    o = nbmaps[a:b, 1].long()
                else:
                    i = nbmaps[a:b, 1].long()
                    o = nbmaps[a:b, 0].long()
                output[o] += torch.mm(input[i], weight[k])
                a += nbsizes[k]

        ctx.for_backwards = (input, weight, nbmaps, nbsizes, transposed)
        return output

    @staticmethod
    @custom_bwd
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        input, weight, nbmaps, nbsizes, transposed = ctx.for_backwards

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)

        if grad_output.device.type == 'cuda':
            sparse_op.convolution_backward_cuda(
                input,
                grad_input,
                grad_output.contiguous(),
                weight,
                grad_weight,
                nbmaps,
                nbsizes.cpu(),
                transposed,
            )
        else:
            raise NotImplementedError
        return grad_input, grad_weight, None, None, None, None, None

