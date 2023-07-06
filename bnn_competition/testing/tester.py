import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

from bnn_competition.dataloaders import BenchmarkDataloader
from bnn_competition.estimators import ComplexityEstimator
from bnn_competition.evaluation import Checker
from bnn_competition.exceptions import NotBinaryException
from bnn_competition.libs.binary_modules import BINARY_MODULES_NAMES
from bnn_competition.tools.metrics import PSNR


class Tester:
    "Testing on Set14 dataset."

    FP_MODELS_INFO = {
        "scalex2_model": {
            "complexity": 1.0,
            "psnr": 33.5951,  # fp model quality on set14
        },
        "scalex4_model": {
            "complexity": 1.0,
            "psnr": 28.4585,  # fp model quality on set14
        },
    }
    EPS_INFO = {
        "scalex2_model": {
            "complexity": 0.01,
            "psnr": 0.16,
        },
        "scalex4_model": {
            "complexity": 0.01,
            "psnr": 0.08,
        },
    }
    INPUT_SHAPE = (1, 3, 96, 96)
    ALPHA = 0.4

    @staticmethod
    def get_device(device: str = None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)
        return device

    def test_model(self, model, scale, device=None):
        device = self.get_device(device)
        model.to(device)
        model.eval()
        dataloader = BenchmarkDataloader(name="set14", scale=scale)
        psnr = PSNR(min_val=0, max_val=255, boundary_size=scale)

        def forward(x, forward_function):
            device = x.device
            x_extended = [
                x.detach().clone(),
                F.hflip(x.detach().clone()),
                F.vflip(x.detach().clone()),
                torch.transpose(x.detach().clone(), 3, 2),
            ]

            y_extended = [forward_function(x.to(device)) for x in x_extended]
            y_extended = [
                y_extended[0],
                F.hflip(y_extended[1]),
                F.vflip(y_extended[2]),
                torch.transpose(y_extended[3], 3, 2),
            ]
            y = torch.cat(y_extended, dim=0).mean(dim=0, keepdim=True)

            return y

        for x, y in tqdm(dataloader.loader, desc=f"testing scalex{scale} model"):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                psnr.update(forward(x, model.forward), y)

        return psnr.compute().item()

    def wrap_model(self, model, modified_modules=None, prefix=""):
        """
        Break down the model (or another module) into modules and change modifiable layers recursively.
        """
        if modified_modules is None:
            modified_modules = set()

        if model not in modified_modules:
            modified_modules.add(model)
            for name, module in model._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name

                if module._get_name() in BINARY_MODULES_NAMES:
                    model._modules[name] = Checker(module, submodule_prefix)

                self.wrap_model(module, modified_modules=modified_modules, prefix=submodule_prefix)
        return model

    def get_score(self, model_complexity, model_psnr, scale):
        fp_psnr = self.FP_MODELS_INFO[f"scalex{scale}_model"]["psnr"]
        eps_acc = self.EPS_INFO[f"scalex{scale}_model"]["psnr"]
        eps_c = self.EPS_INFO[f"scalex{scale}_model"]["complexity"]
        if model_psnr < fp_psnr - eps_acc:
            value = (
                (model_psnr + eps_acc - fp_psnr)
                + (1 - model_complexity) * eps_acc / (1 - eps_c)
                - eps_acc * eps_c / (1 - eps_c)
            )
            return round((1 - eps_c) / eps_acc * value, 4)
        else:
            value = (model_psnr + eps_acc - fp_psnr) + (1 - model_complexity) * eps_acc / eps_c - eps_acc
            return round(eps_c / eps_acc * value, 4)

    def get_model_info(self, model, scale, device=None):
        model_info = {
            "score": 0.0,
            "complexity": self.FP_MODELS_INFO[f"scalex{scale}_model"]["complexity"],
            "psnr": self.FP_MODELS_INFO[f"scalex{scale}_model"]["psnr"],
            "info": "Full precision model was used.",
        }
        if model:
            try:
                model.eval()
                model_complexity = ComplexityEstimator().estimate(
                    model.to(torch.device("cpu")), torch.rand(self.INPUT_SHAPE), f"scalex{scale}_model"
                )
                model_info["complexity"] = model_complexity
                model = self.wrap_model(model)
                if model_complexity <= 1.0:
                    model_psnr = self.test_model(model, scale=scale, device=device)
                    model_score = self.get_score(model_complexity, model_psnr, scale=scale)
                    model_info["info"] = "All checks are passed."
                else:
                    model_info["info"] = f"The model complexity {model_complexity} > 1.0."
                    model_score = 0.0
                    model_psnr = None
            except NotBinaryException as e:
                model_info[
                    "info"
                ] = f"{e.tensor_type} for module {e.module_name} of scalex{scale} model are not binary!!!"
                model_score = 0.0
                model_psnr = None
            if model_score < 0:
                model_info[
                    "info"
                ] = f"The score for scalex{scale} model is equal to {model_score} and was replaced with 0."
                model_score = 0.0
            model_info["score"] = model_score
            model_info["psnr"] = model_psnr
        else:
            model_score = 0.0
        return model_info

    def test(self, scalex2_model=None, scalex4_model=None, device=None):
        scalex2_model_info = self.get_model_info(scalex2_model, scale=2, device=device)
        scalex4_model_info = self.get_model_info(scalex4_model, scale=4, device=device)

        return {
            "score": self.ALPHA * scalex2_model_info["score"] + (1 - self.ALPHA) * scalex4_model_info["score"],
            "scalex2_model": scalex2_model_info,
            "scalex4_model": scalex4_model_info,
        }
