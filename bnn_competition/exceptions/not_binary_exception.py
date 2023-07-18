class NotBinaryException(Exception):
    "Raised when not binary weights or activations found while passing data through binary module"

    def __init__(self, tensor_type, module_name, **kwargs):
        super().__init__(**kwargs)
        self.tensor_type = tensor_type
        self.module_name = module_name
