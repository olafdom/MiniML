from ._core import KNN as KNNEngine, DistanceMetric

class KNN():
    def __init__(self, initial_data: list[list[float]], labels: list[int], metric: str='euclidean'):
        """
        Initializes the given initial data for KNN.
        
        Args:
            initial_data: Initial data for the KNN algorithm
            labels: Labels for the given data
        """
        metric_normalized = metric.strip().lower()
        if metric_normalized == "euclidean":
            self._metric = DistanceMetric.EUCLIDEAN
        elif metric_normalized == "manhattan":
            self._metric = DistanceMetric.MANHATTAN
        else:
            raise ValueError('Given metric is not defined')

        self._cpp_engine = KNNEngine(initial_data, labels, self._metric)

    def predict(self, sample: list[float], k: int=3) -> int:
        """
        Predicts label for the given sample.
        
        Args:
            sample: Single data vector.
            k: Neighbour count.
            
        Returns:
            Predicted label.
        """
        return self._cpp_engine.predict(sample, k)
