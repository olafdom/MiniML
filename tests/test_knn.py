import pytest
import pandas as pd
from pathlib import Path
import miniml as mml

# 1. Fixture do wczytywania danych
# Wczyta plik raz i udostępni dane innym funkcjom
@pytest.fixture
def glass_data():
    """Wczytuje dataset glass.csv i dzieli go na cechy i etykiety."""
    TEST_DIR = Path(__file__).resolve().parent
    file_path = TEST_DIR / ".." / "data" / "glass.csv"

    df = pd.read_csv(file_path)
    initial_data = df.iloc[:, :-1].values.tolist()
    labels = df.iloc[:, -1].tolist()

    return initial_data, labels

# 2. Fixture do inicjalizacji modelu
# Pobiera dane z poprzedniego fixture i tworzy instancję KNN
@pytest.fixture
def knn_model(glass_data):
    """Zwraca wytrenowany model KNN z domyślną metryką."""
    initial_data, labels = glass_data
    return mml.KNN(initial_data, labels)

# 3. Właściwy test (musi zaczynać się od 'test_')
def test_knn_prediction_returns_valid_label(knn_model):
    """Sprawdza, czy model poprawnie zwraca klasę dla nowej próbki."""
    # Przykładowa próbka (ta sama, co w Twoim kodzie)
    sample = [1.67, 10.67, 4.49, 1.1, 80.67, 0.06, 5.75, 0.0, 0.0]
    k = 5

    # Wykonujemy predykcję
    prediction = knn_model.predict(sample, k)

    # PYTESTOWE ASSERTY:
    # Zamiast print(), każemy pytestowi sprawdzić, czy prawda jest prawdziwa.

    # Upewniamy się, że zwrócona wartość to integer (etykieta klasy)
    assert isinstance(prediction, int), "Predykcja powinna być liczbą całkowitą!"

    # Zbiór glass.csv zawiera klasy od 1 do 7 (z brakiem 4),
    # więc upewniamy się, że wynik mieści się w sensownym zakresie.
    assert 1 <= prediction <= 7, f"Zwrócona etykieta {prediction} jest spoza oczekiwanego zakresu klas!"