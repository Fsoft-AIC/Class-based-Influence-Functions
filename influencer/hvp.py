from torch.autograd import grad
import torch

def hvp(y, w, v):
    """ Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.
    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    Raises:
        ValueError: `y` and `w` have a different length.
    """
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length"))
    
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elementwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elementwise_products += torch.sum(grad_elem * v_elem)

    # second grad
    return_grads = grad(elementwise_products, w, create_graph=False)

    return return_grads