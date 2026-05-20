import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import miniml as mml

@pytest.fixture
def glass_data():
    """Wczytuje dataset glass.csv i dzieli go na cechy i etykiety."""
    TEST_DIR = Path(__file__).resolve().parent
    file_path = TEST_DIR / ".." / "data" / "glass.csv"

    df = pd.read_csv(file_path)
    initial_data = df.iloc[:, :-1].values 
    labels = df.iloc[:, -1].values

    return initial_data, labels

@pytest.fixture
def knn_model(glass_data):
    """Zwraca wytrenowany model KNN z domyślną metryką."""
    initial_data, labels = glass_data
    return mml.KNN(initial_data, labels)

def test_knn_prediction_returns_valid_label(knn_model):
    """Sprawdza, czy model poprawnie zwraca klasę dla nowej próbki."""
    sample = [1.67, 10.67, 4.49, 1.1, 80.67, 0.06, 5.75, 0.0, 0.0]
    k = 5

    prediction = knn_model.predict(sample, k)

    assert isinstance(prediction, int), "Predykcja powinna być liczbą całkowitą!"
    assert 1 <= prediction <= 7, f"Zwrócona etykieta {prediction} jest spoza oczekiwanego zakresu klas!"