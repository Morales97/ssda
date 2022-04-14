from __future__ import absolute_import

import torch
from utils.blur_helpers import _get_gaussian_kernel2d, _cast_squeeze_in, _cast_squeeze_out, _get_mean_kernel2d, _get_diagonal_kernel2d, _get_line_kernel2d, _get_hpf_kernel2d
from torch import Tensor
from typing import Tuple, List, Optional
import numbers
from collections.abc import Sequence
from torch.nn.functional import conv2d, pad as torch_pad
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image, to_tensor
from torchvision.utils import save_image
import pdb 

def blur(img: Tensor, kernel_size: List[int], blur_type, sigma: List[float]) -> Tensor:
	"""
		From torch's functional_tensor
		https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional_tensor.py
	"""
	if not (isinstance(img, torch.Tensor)):
		raise TypeError(f"img should be Tensor. Got {type(img)}")

	dtype = img.dtype if torch.is_floating_point(img) else torch.float32
	if blur_type == 'gaussian':
		kernel = _get_gaussian_kernel2d(kernel_size, sigma) # tensor of size (3, 3)
	elif blur_type == 'mean':
		kernel = _get_mean_kernel2d(kernel_size)
	elif blur_type == 'diagonal':
		kernel = _get_diagonal_kernel2d(kernel_size)
	elif blur_type == 'diagonal_flip':
		kernel = _get_diagonal_kernel2d(kernel_size, do_flip=True) 
	elif blur_type == 'horizontal':
		kernel = _get_line_kernel2d(kernel_size)
	elif blur_type == 'vertical':
		kernel = _get_line_kernel2d(kernel_size, do_vertical=True) 
	elif blur_type == 'hpf':
		kernel = _get_hpf_kernel2d(kernel_size)
	else:
		raise Exception('blur type not supported')
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

def mean_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
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


class Blur(torch.nn.Module):
	"""
	Adapted from torch's GaussianBlur
	"""

	def __init__(self, kernel_size, blur_type='gaussian', sigma=(0.1, 2.0)):
		super().__init__()
		#_log_api_usage_once(self)
		self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
		self.blur_type = blur_type
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
		out = _call_blur(img, self.kernel_size, self.blur_type, [sigma, sigma])
		return out

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

def _call_blur(img: Tensor, kernel_size: List[int], blur_type, sigma: Optional[List[float]] = None) -> Tensor:
    """
	adapted from F.gaussian_blur
	https://pytorch.org/vision/main/_modules/torchvision/transforms/functional.html#gaussian_blur
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass #_log_api_usage_once(gaussian_blur)
    if not isinstance(kernel_size, (int, list, tuple)):
        raise TypeError(f"kernel_size should be int or a sequence of integers. Got {type(kernel_size)}")
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if len(kernel_size) != 2:
        raise ValueError(f"If kernel_size is a sequence its length should be 2. Got {len(kernel_size)}")
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError(f"kernel_size should have odd and positive integers. Got {kernel_size}")

    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]

    if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
        raise TypeError(f"sigma should be either float or sequence of floats. Got {type(sigma)}")
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
        sigma = [sigma[0], sigma[0]]
    if len(sigma) != 2:
        raise ValueError(f"If sigma is a sequence, its length should be 2. Got {len(sigma)}")
    for s in sigma:
        if s <= 0.0:
            raise ValueError(f"sigma should have positive values. Got {sigma}")

    t_img = img
    if not isinstance(img, torch.Tensor):
        if not isinstance(img, Image.Image):
            raise TypeError(f"img should be PIL Image or Tensor. Got {type(img)}")

        t_img = pil_to_tensor(img)

    output = blur(t_img, kernel_size, blur_type, sigma)

    if not isinstance(img, torch.Tensor):
        output = to_pil_image(output, mode=img.mode)
    return output

if __name__ == '__main__':

	image = Image.open('/Users/dani/Desktop/sample_img.jpg')
	image = to_tensor(image)

	image_blurred = blur(image, [5,5], 'hpf', [1,1])
	save_image(image_blurred, '/Users/dani/Desktop/sample_blurred.jpg')

	pdb.set_trace()