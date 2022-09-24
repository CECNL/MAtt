import torch
import numpy as np

def symmetric(A):
    size = list(range(len(A.shape)))
    temp = size[-1]
    size.pop()
    size.insert(-1, temp)
    return 0.5 * (A + A.permute(*size))

def is_nan_or_inf(A):
    C1 = torch.nonzero(A == float('inf'))
    C2 = torch.nonzero(A != A)
    if len(C1.size()) > 0 or len(C2.size()) > 0:
        return True
    return False

def is_pos_def(x):
    return torch.all(torch.linalg.eigvals(x) > 0)

def matrix_operator(A, operator):
    u, s, v = A.svd()
    if operator == 'sqrtm':
        s.sqrt_()
    elif operator == 'rsqrtm':
        s.rsqrt_()
    elif operator == 'logm':
        s.log_()
    elif operator == 'expm':
        s.exp_()
    else:
        raise('operator %s is not implemented' % operator)
    
    output = u.mm(s.diag().mm(u.t()))
    
    return output

def tangent_space(A, ref, inverse_transform=False):
    ref_sqrt = matrix_operator(ref, 'sqrtm')
    ref_sqrt_inv = matrix_operator(ref, 'rsqrtm')
    middle = ref_sqrt_inv.mm(A.mm(ref_sqrt_inv))
    if inverse_transform:
        middle = matrix_operator(middle, 'logm')
    else:
        middle = matrix_operator(middle, 'expm')
    out = ref_sqrt.mm(middle.mm(ref_sqrt))
    return out

def untangent_space(A, ref):
    return tangent_space(A, ref, True)

def parallel_transform(A, ref1, ref2):
    print(A.size(), ref1.size(), ref2.size())
    out = untangent_space(A, ref1)
    out = tangent_space(out, ref2)
    return out

def orthogonal_projection(A, B):
    out = A - B.mm(symmetric(B.transpose(0,1).mm(A)))
    return out

def retraction(A, ref):
    data = A + ref
    Q, R = data.qr()
    # To avoid (any possible) negative values in the output matrix, we multiply the negative values by -1
    sign = (R.diag().sign() + 0.5).sign().diag()
    out = Q.mm(sign)
    return out
