""" Util functions for bit packing. """
import numpy as np
import torch


def unpack_bits(tensor: torch.Tensor, dim: int = -1, mask: int = 0b00000001, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    """Unpack bits tensor into bits tensor."""
    return f_unpackbits(tensor, dim=dim, mask=mask, dtype=dtype)


def f_unpackbits(
    tensor: torch.Tensor,
    dim: int = -1,
    mask: int = 0b00000001,
    shape: tuple[int, ...] | None = None,
    out: torch.Tensor | None = None,
    dtype: torch.dtype = torch.uint8,
) -> torch.Tensor:
    """Unpack bits tensor into bits tensor."""
    dim = dim if dim >= 0 else dim + tensor.dim()
    shape_, nibbles, nibble = packshape(tensor.shape, dim=dim, mask=mask, pack=False)
    shape = shape if shape is not None else shape_
    out = out if out is not None else torch.empty(shape, device=tensor.device, dtype=dtype)

    if shape[dim] % nibbles == 0:
        shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype=torch.uint8, device=tensor.device)
        shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
        return torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out=out)

    for i in range(nibbles):
        shift = nibble * i  # type: ignore[assignment]
        sliced_output = tensor_dim_slice(out, dim, slice(i, None, nibbles))
        sliced_input = tensor.narrow(dim, 0, sliced_output.shape[dim])
        torch.bitwise_and(sliced_input >> shift, mask, out=sliced_output)
    return out


def packshape(shape: tuple[int, ...], dim: int = -1, mask: int = 0b00000001, *, pack: bool = True) -> tuple[tuple[int, ...], int, int]:
    """Define pack shape."""
    dim = dim if dim >= 0 else dim + len(shape)
    bits = 8
    nibble = 1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else 0
    nibbles = bits // nibble
    shape = (shape[:dim] + (int(np.ceil(shape[dim] / nibbles)),) + shape[dim + 1 :]) if pack else (shape[:dim] + (shape[dim] * nibbles,) + shape[dim + 1 :])
    return shape, nibbles, nibble


def tensor_dim_slice(tensor: torch.Tensor, dim: int, dim_slice: slice) -> torch.Tensor:
    """Slices a tensor for packing."""
    return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None),) + (dim_slice,)]
