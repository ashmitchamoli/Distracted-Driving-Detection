import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
import os 
from typing import Literal

DATASET_NAME = 'Datasets/annotatedvideosv1/AnnotatedVideos/'
SCENEGRAPH_DIR_NAME = 'sceneGraphs'
IMAGE_EMBEDDING_DIR_NAME = 'imageEmbeddings'

class DrivingDataset(Dataset):
    def __init__(self, videoName : str, split : Literal['train', 'dev', 'test']) -> None:
        super().__init__()

        self.preprocessor = Preprocessor()

        self.data = self.preprocessor.loadAllData(videoName, split)
        self.nodeIndex = self.preprocessor.nodeIndex
        self.edgeAttributeIndex = self.preprocessor.edgeAttributeIndex
        self.classes = set()
        self.imageEmbeddingSize = self.data.iloc[0]['imageEmbedding'].shape[0]
        self.peripheralInputSize = 1 # ! change this later when adding peripheral inputs

        # self.sceneGraphs, self.imageEmbeddings, _ = self.preprocessor.loadAllData(videoName)
        self.X, self.y = self.prepareData()
        self.numClasses = len(self.classes)

    def prepareData(self):
        X = []
        y = []

        def getXY(row):
            X.append([row['SG'], row['imageEmbedding'], torch.tensor([0])]) # ! add peripheral inputs later
            y.append(torch.tensor(row['label'], dtype=torch.long))
            self.classes.add(row['label'])
        
        self.data.apply(getXY, axis=1)

        return X, y

    def __len__(self):
        return len(self.data)
    
    def customCollate(self, batch):
        return batch

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class Preprocessor:
    def __init__(self) -> None:
        pass
    
    def loadSceneGraphData(self, videoName : str, split : str) -> list[Data]:
        """
        videoName : name of the video
        split : 'train', 'dev', 'test'

        Returns a DataFrame of torch_geometric.data.Data objects. Has only one column 'SG'.
        """
        data = self.readSceneGraphs(videoName, split)

        nodeIndex = {}
        edgeAttributeIndex = {}

        def getSceneGraph(row):
            """
            gets a row of the dataframe and appends a corresponding Data object into the scenegraph
            """
            edgeAttr = []
            edgeIndex = [[],
                         []]

            # encode subject nodes
            for node in row['subject']:
                if node not in nodeIndex:
                    nodeIndex[node] = len(nodeIndex)
                edgeIndex[0].append(nodeIndex[node])
            
            # encode object nodes
            for node in row['object']:
                if node not in nodeIndex:
                    nodeIndex[node] = len(nodeIndex)
                edgeIndex[1].append(nodeIndex[node])
            
            # encode edge attributes
            for relation in row['relation']:
                if relation not in edgeAttributeIndex:
                    edgeAttributeIndex[relation] = len(edgeAttributeIndex)
                edgeAttr.append(edgeAttributeIndex[relation])
            
            sceneGraph = Data(edge_index=torch.tensor(edgeIndex, dtype=torch.long),
                              edge_attr=torch.tensor(edgeAttr, dtype=torch.long).reshape(len(edgeIndex[0]), 1),
                              num_nodes=len(nodeIndex),
                              x=torch.ones(len(nodeIndex), 1),
                              num_features=1)

            # sceneGraph.nodeIndex = nodeIndex
            # sceneGraph.edgeAttributeIndex = edgeAttributeIndex

            row['SG'] = sceneGraph

            return row

        data = data.apply(getSceneGraph, axis=1)

        self.nodeIndex = nodeIndex
        self.edgeAttributeIndex = edgeAttributeIndex

        return data[['SG']]

    def readSceneGraphs(self, videoName : str, split : str) -> pd.DataFrame:
        """
        returns a dataframe of scene graphs for the video inside the dataset folder.
        """
        pathToFile = os.path.join(DATASET_NAME, videoName, SCENEGRAPH_DIR_NAME, split + '.json')
        data = pd.read_json(pathToFile, orient='index')

        return data

    def loadImageEmbeddings(self, videoName : str, split : str) -> pd.DataFrame:
        data = pd.read_json(os.path.join(DATASET_NAME, videoName, IMAGE_EMBEDDING_DIR_NAME, split + '.json'), orient='index')
        
        def transform(row):
            row['imageEmbedding'] = torch.tensor(row[list(range(0, len(row)))], dtype=torch.float)

            return row

        data = data.apply(transform, axis=1)

        return data[['imageEmbedding']]

    def readPeripheralInputs(self, videoName : str, split : str):
        pass

    def loadAllData(self, videoName : str, split : str):
        sceneGraphs = self.loadSceneGraphData(videoName, split)
        imageEmbeddings = self.loadImageEmbeddings(videoName, split)
        peripheralInputs = self.readPeripheralInputs(videoName, split)

        def getLabels(row):
            """
            gets the label for each row.
            """
            imageName = row.name.split('/')[-1][:-4]
            label = imageName.split('_')[-1]

            if(label == 'nan'):
                row['label'] = None
            else:
                row['label'] = float(label)

            return row

        # merge the dataframes on the index
        finalDf = sceneGraphs.merge(imageEmbeddings, left_index=True, right_index=True)
        # ! add peripheral inputs

        finalDf = finalDf.apply(getLabels, axis=1)
        finalDf = finalDf.dropna()

        return finalDf
        # return sceneGraphs, imageEmbeddings, peripheralInputs

    def loadFrames(self, videoName : str, split : str):
        # iterate over the frames director
        pass
