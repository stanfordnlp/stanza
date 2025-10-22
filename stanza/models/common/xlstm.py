from omegaconf import OmegaConf
from pprint import pprint
from dacite import from_dict
from dacite import Config as DaciteConfig
import torch

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

def xlstm_stack(input_size):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    xlstm_cfg = f""" 
    mlstm_block:
      mlstm:
        conv1d_kernel_size: 4
        qkv_proj_blocksize: 4
        num_heads: 4
    slstm_block:
      slstm:
        backend: {'cuda' if torch.cuda.is_available() else 'vanilla'} #! only vanilla here works
        num_heads: 4
        conv1d_kernel_size: 4
        bias_init: powerlaw_blockdependent
      feedforward:
        proj_factor: 1.3
        act_fn: gelu
    context_length: 256
    num_blocks: 1
    embedding_dim: %d
    #slstm_at: [1] #[1] # for [] it also works, so if no sLSTM is in the stack
    slstm_at: [0] #[1] # for [] it also works, so if no sLSTM is in the stack
    """ % input_size
    cfg = OmegaConf.create(xlstm_cfg)
    cfg = from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))
    xlstm_stack = xLSTMBlockStack(cfg)
    xlstm_stack = xlstm_stack.to(device=device)
    return xlstm_stack
