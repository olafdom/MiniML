from ._core import KNN as KNNEngine, DistanceMetric
import numpy as np

class KNN():
    def __init__(self, initial_data, labels, metric: str='euclidean'):
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
        np_data = np.asarray(initial_data, dtype=np.float64)
        np_labels = np.asarray(labels, dtype=np.int32)
        self._cpp_engine = KNNEngine(np_data, np_labels, self._metric)

    def predict(self, sample, k: int=3) -> int:
        """
        Predicts label for the given sample.
        
        Args:
            sample: Single data vector.
            k: Neighbour count.
            
        Returns:
            Predicted label.
        """
        np_sample = np.asarray(sample, dtype=np.float64)
        return self._cpp_engine.predict(sample, k)
