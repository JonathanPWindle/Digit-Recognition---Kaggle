import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess():
    trainingData = pd.read_csv("Data/train.csv", dtype=float)
    testData = pd.read_csv("Data/test.csv")

    xTraining = trainingData.drop("label", axis=1).values
    yTraining = trainingData[["label"]].values

    yScaledTraining = []
    for i in range(0,len(yTraining)):
        # print(yTraining[i])
        yScaledTraining.append(np.zeros([10]))
        yScaledTraining[i][int(yTraining[i])] = 1

    xTest = testData.values

    xScaler = MinMaxScaler(feature_range=(0,1))

    xScaledTraining = xScaler.fit_transform(xTraining)


    xScaledTest = xScaler.fit_transform(xTest)

    xBatches = np.array_split(xScaledTraining, len(xScaledTraining) / 50)
    yBatches = np.array_split(yScaledTraining, len(xScaledTraining) / 50)
    testBatches = np.array_split(xScaledTest, len(xScaledTest) / 50)
    return {"xScaledTraining": xScaledTraining, "yScaledTraining": yScaledTraining, "xScaledTest": xScaledTest, \
            "xBatch":xBatches, "yBatch": yBatches, "testBatches": testBatches}




