from types import *
import numpy as np

NUM = ('float','int')
PRIMITIVES = [  fdot
                mat_mul
                ]

# Type Decorators
def type_check(func):
    """ Ensures x is an array of dtype t """=
        def x_type_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
    return x_type_decorator

def function_compatible(func, x_dict, y_dict):
    def get_variables():
        pass
    def shape_compat():
        pass
    def dtype_compat():
        pass
    def accepts_compat():
        pass
    if 'type' in y_dict:
        assert type_compat(y_dict['type'], x_dict['type'], variables), "Types must be compatable"
        t = y_dict['type']
        if t == np.ndarray:
            if 'shape' in y_dict:
                assert shape_compat(), "Shapes must be compatable"
            if 'dtype' in y_dict:
                assert dtype_compat(), "dtype must be compatable"
        if t in (FunctionType, LambdaType):
            if 'accepts' in y_dict:
                assert accepts_compat(), "Inputs must be compatable with function {0}.".format(func)
    else:
        raise KeyError("Type not included.")

def shape_compat():
    pass

@type_check
def mat_poly(x: {'type':np.ndarray,'shape':('{0, int}',),'dtype':NUM},
             W: {'type':np.ndarray,'shape':('{2, int}','{1, int}','{0, int}'), 'dtype':NUM}
             B: {'type':np.ndarray,'shape':('{1, int}',), 'dtype':NUM}):
                -> {'type':np.ndarray,'shape':('{1, int}',), 'dtype':NUM}
    """ Parameters
        ----------
        x : type: np.array, shape:('{0, int}',), dtype: {3, num}
            The input parameter.
        W : type: np.array, shape:('{2, int}','{1, int}','{0, int}'), dtype: {3, num}
            The polynomial weight multiplier, a single 3D array, used like a list of matricies
        B : type: np.array, shape:('{1, int}',), dtype: {3, num}
            The initial bias parameter. Same shape as output.
        {0}: int, input length
        {1}: int, output length
        {2}: int, polynomial order
        """"
        
    out = B
    for i in np.arange(W.shape[0]):
        out += np.dot(W[i,:,:], x**(i+1))
    return out

@type_check
def mat_dot(x: {'type':np.ndarray,'shape':('{0, int}',),'dtype':'{2, type}'},
                W: {'type':np.ndarray,'shape':('{1, int}','{0, int}'),'dtype':'{2, type}'), 
                func1: {'type':FunctionType, 'accepts': ('x','x')},
                func2: {'type':FunctionType, 'accepts': ('x',)}):
                    -> {'type':np.ndarray,'shape':('{1,int}',),'dtype':'{2, type}'}
    out = []
    for i in np.arange(W.shape[0]):
        y = func1(W[i],x)
        y = func2(y)
        out.append(y)
    return np.array(out)
    
@type_check
def mat_compare(x: {'type':np.ndarray,'shape':('{0, int}', '{1, int}'), 'dtype':'{2, type}'},
                y: {'type':np.ndarray,'shape':('{0, int}', '{1, int}'), 'dtype':'{2, type}'},
                func: {'type':FunctionType, 'accepts': ('x','y')})}):
                    -> {'type':np.ndarray,'shape':('{0,int}','{1, int}'),'dtype':'{2, type}'}
    return np.vectorize(func)(x, y)
    
@type_check
def mat_map(x: {'type':np.ndarray, 'shape':'{0, tuple}', 'dtype':'{1, type}'},
              func: {'type':FunctionType, 'accepts': ({'type':'{1, type}'},)}):
                -> {'type':np.ndarray, 'shape':'{0, tuple}', 'dtype':'{1, type}'}
    return np.vectorize(func)(x)
    
@type_check
def mat_reduce(x: {'type':iter, 'dtype':'{0, type}'},
               func: {'type':FunctionType, 'accepts': 'accepts': ({'type':'{0, type}'}, {'type':'{0, type}'})}):
                   -> {'type':'{0, type}'}
    return np.array(reduce(func, x))