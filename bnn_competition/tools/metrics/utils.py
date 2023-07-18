import torch


def convert_to_default_image_range(image, min_val, max_val):
    return torch.clamp((image - min_val) / (max_val - min_val), 0.0, 1.0) * 255.0


def remove_boundary(image, boundary_size):
    return image[..., boundary_size:-boundary_size, boundary_size:-boundary_size]


def get_luminance(image):
    luma_scales = [65.481, 128.553, 24.966]
    luma_scales = image.new_tensor(luma_scales).view(1, 3, 1, 1)
    return (image.mul(luma_scales) / 255.0 + 16.0).sum(dim=1)
