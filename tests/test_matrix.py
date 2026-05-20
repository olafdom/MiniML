import pytest
from miniml import MATRIX

def test_matrix_creation_and_dimensions():
    """Sprawdza, czy macierz poprawnie się tworzy i przechowuje wymiary."""
    mat = Matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert mat.rows() == 2
    assert mat.cols() == 3

def test_matrix_invalid_creation_size():
    """Sprawdza, czy przekazanie złej liczby elementów wyrzuci błąd.
    pybind11 automatycznie tłumaczy std::invalid_argument na Pythonowy ValueError.
    """
    with pytest.raises(ValueError):
        # Macierz 2x2 wymaga 4 elementów, dajemy 3
        Matrix(2, 2, [1.0, 2.0, 3.0])

def test_matrix_addition():
    """Sprawdza dodawanie dwóch macierzy element po elemencie."""
    mat1 = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
    mat2 = Matrix(2, 2, [5.0, 6.0, 7.0, 8.0])

    # Zakładam, że w bindings.cpp przeciążyliście operator '+' dla Pythona
    result = mat1 + mat2

    # Jeśli wystawiliście metodę get_elements() do Pythona:
    assert result.get_elements() == [6.0, 8.0, 10.0, 12.0]

def test_matrix_addition_wrong_dimensions():
    """Sprawdza, czy dodawanie macierzy o różnych wymiarach wywali błąd."""
    mat1 = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
    mat2 = Matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    with pytest.raises(ValueError):
        _ = mat1 + mat2

def test_matrix_transpose():
    """Sprawdza transpozycję macierzy (zamianę wierszy z kolumnami)."""
    # Macierz 2x3
    mat = Matrix(2, 3, [
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    ])

    transposed = mat.transpose()

    assert transposed.rows() == 3
    assert transposed.cols() == 2
    # Wynik powinien być macierzą 3x2
    assert transposed.get_elements() == [
        1.0, 4.0,
        2.0, 5.0,
        3.0, 6.0
    ]

def test_matrix_dot_product():
    """Sprawdza klasyczne mnożenie macierzy (dot product)."""
    # Macierz A (2x3)
    A = Matrix(2, 3, [
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    ])
    # Macierz B (3x2)
    B = Matrix(3, 2, [
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
    ])

    result = A.dot(B)

    assert result.rows() == 2
    assert result.cols() == 2
    # Ręczne obliczenia:
    # row1: [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] -> [58, 64]
    # row2: [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] -> [139, 154]
    assert result.get_elements() == [58.0, 64.0, 139.0, 154.0]

def test_matrix_dot_product_wrong_dimensions():
    """Sprawdza walidację wymiarów przy mnożeniu macierzy."""
    A = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
    B = Matrix(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    with pytest.raises(ValueError):
        # Kolumny A (2) != Wiersze B (3) -> błąd
        _ = A.dot(B)