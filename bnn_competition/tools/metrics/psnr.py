import torchmetrics
from torchmetrics import functional as F

from bnn_competition.tools.metrics.utils import convert_to_default_image_range, get_luminance, remove_boundary


class PSNR(torchmetrics.Metric):
    def __init__(self, min_val=-2.5, max_val=2.5, boundary_size=4, **kwargs):
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.boundary_size = boundary_size

        self.psnr_sum = 0.0
        self.count = 0

    @property
    def full_state_update(self):
        return False

    def reset(self):
        self.psnr_sum = 0.0
        self.count = 0

    def update(self, y_pred, y_true):
        y_true = convert_to_default_image_range(
            remove_boundary(y_true, self.boundary_size), self.min_val, self.max_val
        )
        y_pred = convert_to_default_image_range(
            remove_boundary(y_pred, self.boundary_size), self.min_val, self.max_val
        )

        y_true = get_luminance(y_true).unsqueeze(1)
        y_pred = get_luminance(y_pred).unsqueeze(1)

        self.psnr_sum += F.peak_signal_noise_ratio(y_pred, y_true, data_range=255.0)
        self.count += 1

    def compute(self):
        return self.psnr_sum / self.count
