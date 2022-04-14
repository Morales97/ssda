import torch
from torch import Tensor
from typing import Tuple, List, Optional
import pdb
import torchvision.transforms.functional as TF

'''
	From torch's functional_tensor
	https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional_tensor.py
'''

def _cast_squeeze_in(img: Tensor, req_dtypes: List[torch.dtype]) -> Tuple[Tensor, bool, bool, torch.dtype]:
	need_squeeze = False
	# make image NCHW
	if img.ndim < 4:
		img = img.unsqueeze(dim=0)
		need_squeeze = True

	out_dtype = img.dtype
	need_cast = False
	if out_dtype not in req_dtypes:
		need_cast = True
		req_dtype = req_dtypes[0]
		img = img.to(req_dtype)
	return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype) -> Tensor:
	if need_squeeze:
		img = img.squeeze(dim=0)

	if need_cast:
		if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
			# it is better to round before cast
			img = torch.round(img)
		img = img.to(out_dtype)

	return img

def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
	ksize_half = (kernel_size - 1) * 0.5

	x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
	pdf = torch.exp(-0.5 * (x / sigma).pow(2))
	kernel1d = pdf / pdf.sum()

	return kernel1d


def _get_gaussian_kernel2d(
	kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
	kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
	kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
	kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
	return kernel2d

# DM
def _get_mean_kernel2d(
	kernel_size: List[int]
) -> Tensor:
	kernel2d = torch.ones((kernel_size[0], kernel_size[1]))
	kernel2d /= kernel2d.sum()
	return kernel2d

# DM
def _get_diagonal_kernel2d(
	kernel_size: List[int], do_flip=False
) -> Tensor:
	kernel2d = torch.eye(kernel_size[0])
	if do_flip:
		kernel2d = torch.flip(kernel2d, [0])
	kernel2d /= kernel2d.sum()
	return kernel2d

# DM
def _get_line_kernel2d(
	kernel_size: List[int], do_vertical=False
) -> Tensor:
	kernel2d = torch.zeros(kernel_size)
	if do_vertical:
		kernel2d[:, kernel_size[0]//2] = 1
	else:
		kernel2d[kernel_size[0]//2] = 1
	kernel2d /= kernel2d.sum()
	return kernel2d

def _get_hpf_kernel2d(
	kernel_size: List[int]
) -> Tensor:
	print('High Pass Filter !!')
	kernel2d = torch.ones(kernel_size) * -1
	for i in range(kernel_size[0]):
		if i % 2 == 0:
			kernel2d[i, :] += 1
	for i in range(kernel_size[1]):
		if i % 2 == 0:
			kernel2d[:, i] += 1
	gauss = _get_gaussian_kernel2d([5,5], [1, 1])
	kernel2d = kernel2d * gauss
	kernel2d /= kernel2d.sum()
	return kernel2d


if __name__ == '__main__':
	kernel = _get_gaussian_kernel2d([5,5], [1, 1])
	kernel_mean = _get_mean_kernel2d([3,3])
	kernel_hpf = _get_hpf_kernel2d([5,5])
	pdb.set_trace()