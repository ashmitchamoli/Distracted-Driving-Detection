import numpy as np
import pandas as pd
from lib.driving_dataset.Preprocessor import Preprocessor

preproc = Preprocessor()

def saveSplits(trainDf : pd.DataFrame, 
               devDf : pd.DataFrame, 
               testDf : pd.DataFrame, 
               videoName : str,
               type : str) -> None:
    trainDf.to_json(f'./Datasets/SynDDAnnotated/{videoName}/{type}/train.json',
                    orient='index',)
    devDf.to_json(f'./Datasets/SynDDAnnotated/{videoName}/{type}/dev.json',
                  orient='index')
    testDf.to_json(f'./Datasets/SynDDAnnotated/{videoName}/{type}/test.json',
                   orient='index')

def splitData(videoName : str) -> None:
    sceneGraphData = pd.read_json(f'./Datasets/SynDDAnnotated/{videoName}/sceneGraphs.json', orient='index')
    imageEmbeddings = pd.read_json(f'./Datasets/SynDDAnnotated/{videoName}/imageEmbeddings.json', orient='index')

    allData = sceneGraphData.merge(imageEmbeddings, left_index=True, right_index=True)

    sceneGraphData = allData[sceneGraphData.columns]
    imageEmbeddings = allData[imageEmbeddings.columns]

    print(sceneGraphData.shape)
    print(imageEmbeddings.shape)

    # split into train, test, dev
    trainSize = 0.6
    devSize = 0.2
    testSize = 1 - trainSize - devSize

    np.random.seed(42)

    # get indices
    indicesSet = set(range(len(sceneGraphData)))

    trainIndices = set(np.random.choice(list(indicesSet), int(trainSize * len(sceneGraphData)), replace=False))
    indicesSet.difference_update(trainIndices)

    devIndices = set(np.random.choice(list(indicesSet), int(devSize * len(sceneGraphData)), replace=False))
    indicesSet.difference_update(devIndices)

    testIndices = np.array(list(indicesSet))
    trainIndices = np.array(list(trainIndices))
    devIndices = np.array(list(devIndices))

    # get data
    sceneGraphTrain = sceneGraphData.iloc[trainIndices]
    sceneGraphDev = sceneGraphData.iloc[devIndices]
    sceneGraphTest = sceneGraphData.iloc[testIndices]

    imageEmbeddingsTrain = imageEmbeddings.iloc[trainIndices]
    imageEmbeddingsDev = imageEmbeddings.iloc[devIndices]
    imageEmbeddingsTest = imageEmbeddings.iloc[testIndices]

    # save
    saveSplits(sceneGraphTrain, sceneGraphDev, sceneGraphTest, videoName, 'sceneGraphs')
    saveSplits(imageEmbeddingsTrain, imageEmbeddingsDev, imageEmbeddingsTest, videoName, 'imageEmbeddings')

if __name__ == '__main__':
    splitData('test_video')