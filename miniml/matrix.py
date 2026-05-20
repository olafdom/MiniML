from . import _core
import numpy as np

class MATRIX():
    def __init__(self, data):
        """
        Initializing Matrix.
        """
        np_data = np.asarray(data, dtype=np.float64)
        
        if np_data.ndim != 2:
            raise ValueError(f"Data must be a 2 dimension array. Given {np_data.ndim}.")
            
        self._cpp_matrix = _core._Matrix(np_data)

    @classmethod
    def _from_backend(cls, cpp_matrix_instance):
        """
        Creates Python MATRIX when given cpp matrix instance
        """
        obj = cls.__new__(cls) 
        obj._cpp_matrix = cpp_matrix_instance 
        
        return obj
    
    def rows(self):
        return self._cpp_matrix.rows()

    def cols(self):
        return self._cpp_matrix.cols()
    
    def transpose(self):
        raw_cpp_result = self._cpp_matrix.transpose()
        
        return self._from_backend(raw_cpp_result)
    
    def dot(self, other):
        if not isinstance(other, MATRIX):
            raise TypeError("Możesz mnożyć (dot) tylko z innym obiektem Matrix.")
            
        raw_cpp_result = self._cpp_matrix.dot(other._cpp_matrix)
        
        return self._from_backend(raw_cpp_result)

    def __add__(self, other):
        if not isinstance(other, MATRIX):
            raise TypeError("Invalid type. You can only add a matrix to a matrix.")
            
        raw_cpp_result = self._cpp_matrix.__add__(other._cpp_matrix)
        
        return self._from_backend(raw_cpp_result)
    
    def get_elements(self):
        """Returns the flattened matrix elements as a list."""
        return self._cpp_matrix.get_elements()