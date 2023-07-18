import torch
from torchvision import transforms

from bnn_competition.dataloaders.datasets.superresolution_dataset import SuperResolutionDataset
from bnn_competition.dataloaders.datasets_info import DATASETS_PATHS
from bnn_competition.dataloaders.transforms import CoupledRandomRectangularRotation


class DIV2KDataloader:
    """
    Dataloader for DIV2K dataset.
    """

    def __init__(
        self,
        batch_size: int,
        patch_size: int,
        scale: int,
        num_workers: int = 8,
    ):
        """
        Args:
            patch_size (int): Size for random crop augmentation.
            scale (int): Upscaling factor.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.patch_size = patch_size
        self.scale = scale

        self.train_transforms = transforms.Compose([CoupledRandomRectangularRotation()])
        self.val_transforms = None

        self._train_loader = None
        self._val_loader = None

    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        if not self._train_loader:
            dataset = SuperResolutionDataset(
                DATASETS_PATHS["div2k_train"],
                scale=self.scale,
                patch_size=self.patch_size,
                transforms=self.train_transforms,
                train=True,
            )

            self._train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.num_workers,
            )
        return self._train_loader

    @property
    def val_loader(self) -> torch.utils.data.DataLoader:
        if not self._val_loader:
            dataset = SuperResolutionDataset(
                DATASETS_PATHS["div2k_val"],
                scale=self.scale,
                transforms=self.val_transforms,
                train=False,
            )

            self._val_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
            )
        return self._val_loader
