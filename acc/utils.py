import torch

def to_int64(tensor: torch.Tensor, log_sf: int):
    tensor_ = tensor.to(torch.float64)
    tensor_ = torch.round(tensor_ * (1 << log_sf)).to(torch.int64)
    return tensor_

def to_float(tensor: torch.Tensor, log_sf: int, to_type: torch.dtype = torch.float32):
    tensor_ = (tensor / (1 << log_sf)).to(to_type)
    return tensor_

def rescale(tensor: torch.Tensor, log_sf: int):
    assert tensor.dtype == torch.int64
    tensor_abs = tensor.abs()
    tensor_abs += (1 << (log_sf - 1))
    tensor_abs >>= log_sf
    tensor = tensor.sign() * tensor_abs
    return tensor

# kill ours
def fromto_int64(tensor: torch.Tensor, log_sf: int):
    return to_float(to_int64(tensor, log_sf), log_sf)

def compare_q(t: torch.Tensor, t_q: torch.Tensor, log_sf: int):
    t_ = to_float(t_q, log_sf, torch.float64)
    return (t - t_).abs().max().item(), (t - t_).abs().mean().item()