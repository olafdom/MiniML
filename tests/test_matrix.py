import pytest
from miniml import MATRIX

def test_matrix_creation_and_dimensions():
    """Checks if the matrix is created correctly and stores its dimensions."""
    mat = MATRIX([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert mat.rows() == 2
    assert mat.cols() == 3

def test_matrix_invalid_creation_dimensions():
    """Checks if passing a non-2D array raises a ValueError."""
    with pytest.raises(ValueError):
        # Passing a 1D list instead of a 2D list/array
        MATRIX([1.0, 2.0, 3.0])

def test_matrix_addition():
    """Checks element-wise addition of two matrices."""
    mat1 = MATRIX([[1.0, 2.0], [3.0, 4.0]])
    mat2 = MATRIX([[5.0, 6.0], [7.0, 8.0]])

    result = mat1 + mat2

    assert result.get_elements() == [6.0, 8.0, 10.0, 12.0]

def test_matrix_addition_wrong_dimensions():
    """Checks if adding matrices of different dimensions raises an error."""
    mat1 = MATRIX([[1.0, 2.0], [3.0, 4.0]])
    mat2 = MATRIX([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Depending on your C++ implementation, this might raise ValueError or RuntimeError.
    # pybind11 translates std::invalid_argument to ValueError.
    with pytest.raises(ValueError):
        _ = mat1 + mat2

def test_matrix_transpose():
    """Checks matrix transposition (swapping rows with columns)."""
    mat = MATRIX([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    transposed = mat.transpose()

    assert transposed.rows() == 3
    assert transposed.cols() == 2
    
    # The result should be flattened row by row
    assert transposed.get_elements() == [
        1.0, 4.0,
        2.0, 5.0,
        3.0, 6.0
    ]

def test_matrix_dot_product():
    """Checks standard matrix multiplication (dot product)."""
    A = MATRIX([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    
    B = MATRIX([
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0]
    ])

    result = A.dot(B)

    assert result.rows() == 2
    assert result.cols() == 2
    
    # Manual calculation:
    # row1: [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] -> [58, 64]
    # row2: [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] -> [139, 154]
    assert result.get_elements() == [58.0, 64.0, 139.0, 154.0]

def test_matrix_dot_product_wrong_dimensions():
    """Checks dimension validation during matrix multiplication."""
    A = MATRIX([[1.0, 2.0], [3.0, 4.0]])
    B = MATRIX([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    with pytest.raises(ValueError):
        # A cols (2) != B rows (3) -> should raise an error
        _ = A.dot(B)