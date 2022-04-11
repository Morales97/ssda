from __future__ import absolute_import

import torch
from utils.blur_helpers import *
from torch import Tensor
from typing import Tuple, List, Optional



def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
	"""
		From torch's functional_tensor
		https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional_tensor.py
	"""
	if not (isinstance(img, torch.Tensor)):
		raise TypeError(f"img should be Tensor. Got {type(img)}")

	dtype = img.dtype if torch.is_floating_point(img) else torch.float32
	kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
	kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

	img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(
		img,
		[
			kernel.dtype,
		],
	)

	# padding = (left, right, top, bottom)
	padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
	img = torch_pad(img, padding, mode="reflect")
	img = conv2d(img, kernel, groups=img.shape[-3])

	img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
	return img


class GaussianBlur(torch.nn.Module):
	"""Blurs image with randomly chosen Gaussian blur.
	If the image is torch Tensor, it is expected
	to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

	Args:
		kernel_size (int or sequence): Size of the Gaussian kernel.
		sigma (float or tuple of float (min, max)): Standard deviation to be used for
			creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
			of float (min, max), sigma is chosen uniformly at random to lie in the
			given range.

	Returns:
		PIL Image or Tensor: Gaussian blurred version of the input image.

	"""

	def __init__(self, kernel_size, sigma=(0.1, 2.0)):
		super().__init__()
		_log_api_usage_once(self)
		self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
		for ks in self.kernel_size:
			if ks <= 0 or ks % 2 == 0:
				raise ValueError("Kernel size value should be an odd and positive number.")

		if isinstance(sigma, numbers.Number):
			if sigma <= 0:
				raise ValueError("If sigma is a single number, it must be positive.")
			sigma = (sigma, sigma)
		elif isinstance(sigma, Sequence) and len(sigma) == 2:
			if not 0.0 < sigma[0] <= sigma[1]:
				raise ValueError("sigma values should be positive and of the form (min, max).")
		else:
			raise ValueError("sigma should be a single number or a list/tuple with length 2.")

		self.sigma = sigma

	@staticmethod
	def get_params(sigma_min: float, sigma_max: float) -> float:
		"""Choose sigma for random gaussian blurring.

		Args:
			sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
			sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

		Returns:
			float: Standard deviation to be passed to calculate kernel for gaussian blurring.
		"""
		return torch.empty(1).uniform_(sigma_min, sigma_max).item()

	def forward(self, img: Tensor) -> Tensor:
		"""
		Args:
			img (PIL Image or Tensor): image to be blurred.

		Returns:
			PIL Image or Tensor: Gaussian blurred image
		"""
		sigma = self.get_params(self.sigma[0], self.sigma[1])
		return gaussian_blur(img, self.kernel_size, [sigma, sigma])

	def __repr__(self) -> str:
		s = f"{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma})"
		return s


	def _setup_size(size, error_msg):
		if isinstance(size, numbers.Number):
			return int(size), int(size)

		if isinstance(size, Sequence) and len(size) == 1:
			return size[0], size[0]

		if len(size) != 2:
			raise ValueError(error_msg)

		return size

