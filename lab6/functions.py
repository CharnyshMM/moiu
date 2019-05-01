import numpy as np 

def foreach_predicate(iterable, predicate):
    for i, item in enumerate(iterable):
        if not predicate(item):
            return i, item
    return -1, None

def get_A_b_matrix(A_matrix, J_b):
    result = None
    for j in J_b:
        s = A_matrix[:,j:j+1]
        if result is None:
            result = s
        else:
            result = np.append(result, s, axis=1)
    return result


def get_c_b(c, J_b):
    # c is expected to be of type np.array (flat)
    ar = [c[i] for i in J_b]
    return np.matrix(ar)