import random


class SRRandomCrop:
    """
    Random crop for super-resolution task.
    Differences from `transforms.RandomCrop`:
        - Takes low and high resolution images as input.
        - Crops for both images are the same.
    """

    def __init__(self, patch_size, scale):
        self.patch_size = patch_size
        self.scale = scale

    @staticmethod
    def crop(image, start_h, start_w, patch_size):
        return image[start_h : start_h + patch_size, start_w : start_w + patch_size, ...]

    def __call__(self, low_resolution_image, high_resolution_image):
        h, w = low_resolution_image.shape[:2]

        start_h = random.randrange(0, h - self.patch_size // self.scale + 1)
        start_w = random.randrange(0, w - self.patch_size // self.scale + 1)

        low_resolution_image = self.crop(low_resolution_image, start_h, start_w, self.patch_size // self.scale)
        high_resolution_image = self.crop(
            high_resolution_image, start_h * self.scale, start_w * self.scale, self.patch_size
        )

        return low_resolution_image, high_resolution_image
