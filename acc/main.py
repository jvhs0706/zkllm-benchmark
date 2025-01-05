import os, sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import math

from tqdm import tqdm

from gptq_data_utils import *

parser = argparse.ArgumentParser(description='LLaMa-2 Self-Attention')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use.')
parser.add_argument('seq_len', type=int, help='The sequence length to use for self-attn')
parser.add_argument('nsamples', type=int, help='The number of samples tested')

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *

VALUE_LOGSF = 12
ACCU_LOGSF = 12

# def softmax_q(X: torch.Tensor, dim: int):
#     assert X.dtype == torch.int64
#     X_ = to_float(X, VALUE_LOGSF, torch.float64)
#     X_ = torch.log(torch.exp(X_).sum(axis = -1))

def rmsnorm_q(X: torch.Tensor, rmsnorm: nn.Module):
    assert X.dtype == torch.int64 
    X_ = to_float(X, VALUE_LOGSF, torch.float64)
    variance = X_.pow(2).mean(-1, keepdim=True)
    RINV = to_int64(torch.rsqrt(variance + rmsnorm.variance_epsilon), ACCU_LOGSF)
    X = rescale(X * RINV, ACCU_LOGSF)
    W = to_int64(rmsnorm.weight, ACCU_LOGSF)
    return rescale(W * X, ACCU_LOGSF)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def get_rotary_emb(seq_len, embed_dim, attn):
    cos, sin = attn.rotary_emb(torch.randn(1, seq_len, embed_dim, device = 0), torch.arange(seq_len, device = 0).unsqueeze(0))
    return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    q_q, k_q = to_int64(q, VALUE_LOGSF), to_int64(k, VALUE_LOGSF)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    cos_q = to_int64(cos, ACCU_LOGSF)
    sin_q = to_int64(sin, ACCU_LOGSF)
    q_embed = rescale((q_q * cos_q) + (rotate_half(q_q) * sin_q), ACCU_LOGSF)
    k_embed = rescale((k_q * cos_q) + (rotate_half(k_q) * sin_q), ACCU_LOGSF)
    return to_float(q_embed, VALUE_LOGSF), to_float(k_embed, VALUE_LOGSF)
    
def matmul_q(A: torch.Tensor, B: torch.Tensor, log_sf):
    A_ = fromto_int64(A, log_sf)
    B_ = fromto_int64(B, log_sf)
    return fromto_int64(A_ @ B_, log_sf)

def forward_q(X, fc, log_sf):
    return matmul_q(X, fc.weight.T, log_sf)
    

def attn_q(X: torch.Tensor, attn: nn.Module, position_ids):
    assert attn.num_key_value_groups == 1
    assert X.dtype == torch.int64
    seq_len, embed_dim = X.shape
    X_ = to_float(X, VALUE_LOGSF, torch.float32)
    Q = forward_q(X_, attn.q_proj, VALUE_LOGSF).view(-1, attn.num_heads, attn.head_dim).transpose(0, 1)
    K = forward_q(X_, attn.k_proj, VALUE_LOGSF).view(-1, attn.num_heads, attn.head_dim).transpose(0, 1)
    V = forward_q(X_, attn.v_proj, VALUE_LOGSF).view(-1, attn.num_heads, attn.head_dim).transpose(0, 1)

    cos, sin = attn.rotary_emb(V[None, :], position_ids)
    
    Q, K = apply_rotary_pos_emb(Q[None, :], K[None, :], cos, sin)
    Q.squeeze_(0)
    K.squeeze_(0)
    A_ = Q @ K.transpose(-2, -1)
    A = to_int64(A_, ACCU_LOGSF)

    # an upper triangular mask for perplexity
    mask = torch.triu(torch.ones(seq_len, seq_len, device = 0, dtype = bool), diagonal = 1)

    A -= torch.max(A * ~mask, dim = -1, keepdim = True).values 

    shift = math.sqrt(attn.head_dim) * torch.log((torch.exp((to_float(A, ACCU_LOGSF) / math.sqrt(attn.head_dim))) * ~mask).sum(axis = -1, keepdim = True))
    shift = to_int64(shift, ACCU_LOGSF)
    A -= shift
    attn_output = (torch.exp(to_float(A, ACCU_LOGSF, torch.float64) / math.sqrt(attn.head_dim)).float()) * ~mask

    attn_output = matmul_q(attn_output, V, VALUE_LOGSF)

    attn_output = attn_output.transpose(0, 1).contiguous()
    attn_output = attn_output.view(seq_len, embed_dim)
    attn_output = forward_q(attn_output, attn.o_proj, VALUE_LOGSF)
    
    return to_int64(attn_output, VALUE_LOGSF)

def ffn_q(X: torch.Tensor, ffn: nn.Module):
    X_ = to_float(X, VALUE_LOGSF, torch.float32)
    GATE_ = forward_q(X_, ffn.gate_proj, VALUE_LOGSF)
    ACT = to_int64(ffn.act_fn(GATE_), VALUE_LOGSF)
    UP = to_int64(forward_q(X_, ffn.up_proj, VALUE_LOGSF), VALUE_LOGSF)   
    MID = rescale(ACT * UP, VALUE_LOGSF)
    MID_ = to_float(MID, VALUE_LOGSF, torch.float32)
    return to_int64(forward_q(MID_, ffn.down_proj, VALUE_LOGSF), VALUE_LOGSF)

def layer_q(X_q_in: torch.Tensor, layer: nn.Module):
    X_q_in_norm = rmsnorm_q(X_q_in, layer.input_layernorm)
    X_q_attn = attn_q(X_q_in_norm, layer.self_attn, torch.arange(args.seq_len, device = 0).unsqueeze(0))
    X_q_mid = X_q_in + X_q_attn
    X_q_mid_norm = rmsnorm_q(X_q_mid, layer.post_attention_layernorm)
    X_q_out = ffn_q(X_q_mid_norm, layer.mlp)
    X_q_out += X_q_mid
    return X_q_out


if __name__ == '__main__':
    # completely disable gradient computation
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    model_card = f"meta-llama/Llama-2-{args.model_size}b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only = True, cache_dir = "../../zkllm-ccs2024/model-storage", attn_implementation="eager")
    layers = model.model.layers
    head = model.lm_head
    norm = model.model.norm
    head.cuda()
    norm.cuda()

    model.model.layers.cpu()

    input_tok = get_test_tokens('c4', 0, args.seq_len, f"meta-llama/Llama-2-{args.model_size}b-hf")
    nsamples = min(input_tok.numel() // args.seq_len, args.nsamples)
    input_tok = input_tok[0, :(args.seq_len * nsamples)].view(nsamples, args.seq_len)
    emb = model.model.embed_tokens(input_tok)

    # attention_mask = model.model._prepare_decoder_attention_mask(
    #         torch.ones(1, args.seq_len, dtype=torch.bool),
    #         (1, args.seq_len), emb[0:1], 0).to(0)

    loss_fct = torch.nn.CrossEntropyLoss().cuda()

    # get the gpu usage of 0
    gpu_usage = torch.cuda.memory_allocated(0) / 1024**3

    loss, loss_, loss_comp = 0, 0, 0

    X, X_q = emb.clone(), to_int64(emb, VALUE_LOGSF)
    casual_mask = model.model._update_causal_mask(None, X[0:1].cuda(), torch.arange(0, args.seq_len, device=0), None, None)


    for i in tqdm(range(len(layers))):
        layers[i].cuda()
        for idx in range(nsamples):
            (X[idx:idx+1, :], ) = layers[i](X[None, idx].cuda(), position_ids = torch.arange(args.seq_len, device = 0).unsqueeze(0), attention_mask = casual_mask)
            X_q[idx] = layer_q(X_q[idx].cuda(), layers[i])
        layers[i].cpu()
    
    for idx in range(nsamples):
        logits = head(norm(X[idx].cuda()))
        logits_ = head(norm(to_float(X_q[idx].cuda(), VALUE_LOGSF)))

        shifted_logits = logits[:-1, :].contiguous()
        shifted_logits_ = logits_[:-1, :].contiguous()
        # shifted_logits_comp = logits_comp[:-1, :].contiguous().cuda()
        shifted_labels = input_tok[idx, 1:].cuda()

        loss += loss_fct(shifted_logits, shifted_labels).exp().item()
        loss_ += loss_fct(shifted_logits_, shifted_labels).exp().item()
        # loss_comp += loss_fct(shifted_logits_comp, shifted_labels).exp().item()
        
    print(f"PPL changed from {loss / nsamples} to {loss_ / nsamples}, increasing by {(loss_ - loss) / nsamples}.")
    if loss_ < loss:
        print(f"The quantization error is on our side!")
    