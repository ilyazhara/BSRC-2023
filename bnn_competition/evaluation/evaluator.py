import torch
import torchvision.transforms.functional as F
from prettytable import PrettyTable

from bnn_competition.dataloaders import AVAILABLE_BENCHMARKS, BenchmarkDataloader
from bnn_competition.estimators import ComplexityEstimator
from bnn_competition.evaluation.checker import Checker
from bnn_competition.exceptions import NotBinaryException
from bnn_competition.libs.binary_modules import BINARY_MODULES_NAMES
from bnn_competition.tools.metrics import PSNR


class Evaluator:
    FP_MODELS_INFO = {
        "scalex2_model": {
            "complexity": 1.0,
            "set5": 38.0601,
            "set14": 33.5951,
            "b100": 32.1903,
            "urban100": 32.0406,
        },
        "scalex4_model": {
            "complexity": 1.0,
            "set5": 31.9869,
            "set14": 28.4585,
            "b100": 27.4632,
            "urban100": 25.6885,
        },
    }
    INPUT_SHAPE = (1, 3, 96, 96)

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

    def evaluate_model(self, model, scale, dataset_name, device=None):
        device = self.get_device(device)
        model.to(device)
        model.eval()
        if scale in [2, 4]:
            dataloader = BenchmarkDataloader(name=dataset_name, scale=scale)
            psnr = PSNR(min_val=0, max_val=255, boundary_size=scale)
        else:
            raise ValueError(f"Scale can be only 2 or 4, got scale={scale}.")

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

        for x, y in dataloader.loader:
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

    def print_evaluation_results(self, model, model_name, scale, device=None):
        model.eval()
        complexity = ComplexityEstimator().estimate(
            model.to(torch.device("cpu")), torch.rand(self.INPUT_SHAPE), model_name
        )
        model = self.wrap_model(model)

        table = PrettyTable(["Model"] + ["complexity"] + AVAILABLE_BENCHMARKS)
        table.print_empty = False
        table.float_format = ".4"

        row = [model_name]
        row.append(self.FP_MODELS_INFO[model_name]["complexity"])
        for name in AVAILABLE_BENCHMARKS:
            row.append(self.FP_MODELS_INFO[model_name][name])
        table.add_row(row)

        row = ["binary " + model_name]
        row.append(complexity)
        for name in AVAILABLE_BENCHMARKS:
            row.append(self.evaluate_model(model, scale, name, device=device))
        table.add_row(row)

        print(table)

    def evaluate(self, scalex2_model=None, scalex4_model=None, device=None):
        if scalex2_model:
            try:
                self.print_evaluation_results(scalex2_model, "scalex2_model", scale=2, device=device)
            except NotBinaryException as e:
                print(f"{e.tensor_type} for module {e.module_name} of scalex2 model are not binary!!!")
        if scalex4_model:
            try:
                self.print_evaluation_results(scalex4_model, "scalex4_model", scale=4, device=device)
            except NotBinaryException as e:
                print(f"{e.tensor_type} for module {e.module_name} of scalex4 model are not binary!!!")
