import torch
import torchvision.transforms.functional as F


class CoupledRandomRectangularRotation:
    """
    Rectangular rotation for image-to-image task.
    Rotates image in 0, 90, 180, or 270 degrees.
    Takes raw and target resolution images as input.
    Rotates both images for the same angle.
    """

    def __call__(self, data):
        raw_image, target_image = data

        if torch.rand(1) < 0.5:
            raw_image = F.vflip(raw_image)
            raw_image = torch.permute(raw_image, (0, 2, 1))

            target_image = F.vflip(target_image)
            target_image = torch.permute(target_image, (0, 2, 1))

        return raw_image, target_image
