import miniml as mml
import pandas as pd

# data from: https://www.kaggle.com/datasets/uciml/glass 
file_path = "../data/glass.csv"
df = pd.read_csv(file_path)

initial_data = df.iloc[:, :-1].values.tolist()
labels = df.iloc[:, -1].tolist()

knn = mml.KNN(initial_data, labels)

print(knn.predict([1.67,10.67,4.49,1.1,80.67,0.06,5.75,0,0], 5))