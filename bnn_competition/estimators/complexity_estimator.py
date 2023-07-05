import torch
from torchprofile import profile_macs

from bnn_competition.estimators.hooks import DataSaverHook
from bnn_competition.estimators.utils import evaluate_binary_macs
from bnn_competition.libs.binary_modules import BINARY_MODULES_NAMES


class ComplexityEstimator:
    """
    Class which estimates complexity.
    """

    FP_MODEL_COMPLEXITY = {
        "scalex2_model": 12649955328.0,
        "scalex4_model": 12975390720.0,
    }

    def get_output_shapes(self, model, input_data):
        output_shapes = {}
        data_saver = {}
        handles = {}
        for name, module in model.named_modules():
            if module._get_name() in BINARY_MODULES_NAMES:
                data_saver[name] = DataSaverHook(store_output=True)
                handles[name] = module.register_forward_hook(data_saver[name])

        with torch.no_grad():
            model(input_data)

        for name, module in model.named_modules():
            if module._get_name() in BINARY_MODULES_NAMES:
                if data_saver[name].output is not None:
                    output_shapes[name] = data_saver[name].output[0].unsqueeze(0).shape
                handles[name].remove()

        return output_shapes

    def estimate(self, model, input_data, model_name):
        """
        `model_name` is either 'scalex4_model' or 'scalex2_model'.
        """

        output_shapes = self.get_output_shapes(model, input_data)

        model.eval()
        total_macs = profile_macs(model, torch.rand((1,) + input_data.shape[1:]))

        binary_macs = 0.0
        for name, module in model.named_modules():
            if module._get_name() in BINARY_MODULES_NAMES and output_shapes.get(name):
                binary_macs += evaluate_binary_macs(module, output_shapes[name])

        return ((total_macs - binary_macs) + binary_macs / 8) / self.FP_MODEL_COMPLEXITY[model_name]
