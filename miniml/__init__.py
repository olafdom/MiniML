# Importujemy Matrix bezpośrednio ze skompilowanego przez CMake modułu C++ (_core)
from ._core import Matrix

# Importujemy Wasz wrapper dla KNN z pliku knn.py
from .knn import KNN

# Definiujemy, co ma być widoczne publicznie po wpisaniu `import miniml`
__all__ = ["Matrix", "KNN"]