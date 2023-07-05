def get_shape(x, dims=None):
    """
    Return shape based on given tensor to quantize and dims to keep

    Args:
        x: Tensor to quantize.
        dims: Dimensions to keep in step shape.
    """
    if dims is None:
        return None
    return list(x.shape[i] if i in dims else 1 for i in range(x.dim()))
