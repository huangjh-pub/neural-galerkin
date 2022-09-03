
def get_basis(name, **kwargs):
    """
    Get a basis instance given keyword arguments
    :param name: name of the basis
    :return: basis instance
    """
    if name == "BezierTensorBasis":
        from .bezier_tensor import BezierTensorBasis
        return BezierTensorBasis(**kwargs)
    else:
        raise NotImplementedError
