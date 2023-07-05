import torch

from bnn_competition.dataloaders.datasets.superresolution_dataset import SuperResolutionDataset
from bnn_competition.dataloaders.datasets_info import AVAILABLE_BENCHMARKS, DATASETS_PATHS


class BenchmarkDataloader:
    """
    Dataloader for benchmark super-resolution datasets (shadowset).
    """

    def __init__(
        self,
        name,
        scale: int,
        num_workers: int = 8,
    ):
        """
        Args:
            name (str): Name of dataset.
            scale (int): Upscaling factor.
        """
        assert name in AVAILABLE_BENCHMARKS, "Expected name to be one of {}, found {}.".format(
            AVAILABLE_BENCHMARKS, name
        )

        self.name = name
        self.scale = scale
        self.num_workers = num_workers

        self.transforms = None
        self._loader = None

    @property
    def loader(self) -> torch.utils.data.DataLoader:
        if not self._loader:
            dataset = SuperResolutionDataset(
                path=DATASETS_PATHS[self.name], scale=self.scale, transforms=self.transforms
            )

            self._loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
            )
        return self._loader
