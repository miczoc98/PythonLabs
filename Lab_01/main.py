import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as skl

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

dataset = load_boston()
df = pd.DataFrame(data = dataset["data"], columns = dataset["feature_names"])
df["MEDV"] = dataset["target"]

print (df.head(5))


df.hist(column = "RM", bins = range(3, 10))
plt.savefig("RMhist.png")

df.hist(column = "TAX")
plt.savefig("TAXhist.png")

data_types = ["RM", "TAX"]
target = "MEDV"
lr_model = LinearRegression()


for data_type in data_types:
    df.plot(kind = "scatter", x = data_type, y = target)

    data_as_row = np.array(df[data_type]).reshape(-1, 1)
    lr_model.fit(data_as_row, df[target])

    predict = lr_model.predict(data_as_row)
    plt.plot(df[data_type], predict, c = "red")

    score = lr_model.score(data_as_row, df[target])
    print("R-squared value of " + data_type + " fit : " + str(score))

    plt.save(data_type + "scatter.png")
