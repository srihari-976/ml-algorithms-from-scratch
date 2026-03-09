from sklearn.datasets import make_circles
import pandas as pd

X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)

df = pd.DataFrame(X, columns=["data1","data2"])
df["output"] = y

print(df.head())

df.to_csv("binary_polynomial_dataset.csv", index=False)