class DataSaverHook:
    """
    Forward hook that stores the input and output of a module
    """

    def __init__(self, store_input: bool = False, store_output: bool = False):
        """
        Args:
            store_input (bool, optional): If True, stores input to the module. Defaults to False.
            store_output (bool, optional): If True, stores output to the module. Defaults to False.
        """
        self.store_input = store_input
        self.store_output = store_output

        self.input = None
        self.output = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input = input_batch

        if self.store_output:
            self.output = output_batch
