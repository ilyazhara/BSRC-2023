# dict which contains the position of input channel dimension (C_in) for different types of layers
IN_CHANNEL_DIM = {
    "Linear": 1,  # weight tensor has shape [C_out, C_in]
    "Conv2d": 1,  # weight tensor has shape [C_out, C_in, K, K]
    "ConvTranspose2d": 0,  # weight tensor has shape [C_in, C_out, K, K]
}

# dict which contains the position of output channel dimension (C_out) for different types of layers
OUT_CHANNEL_DIM = {
    "Linear": 0,  # weight tensor has shape [C_out, C_in]
    "Conv2d": 0,  # weight tensor has shape [C_out, C_in, K, K]
    "ConvTranspose2d": 1,  # weight tensor has shape [C_in, C_out, K, K]
}
