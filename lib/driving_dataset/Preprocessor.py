import pandas as pd
from torch_geometric.data import Data
import torch
import os 

DATASET_NAME = 'Datasets/DDDatasetAnnotated'
SCENEGRAPH_FILE_NAME = 'sceneGraphs.json'
IMAGE_EMBEDDING_FILE_NAME = 'imageEmbeddings.json'

class Preprocessor:
    def __init__(self) -> None:
        pass

    def loadSceneGraphData(self, videoName : str) -> list[Data]:
        """
        dataPath : path to scene graph json file

        Returns a DataFrame of torch_geometric.data.Data objects.
        """
        data = self.readSceneGraphs(videoName)

        # sceneGraphs = []

        def getSceneGraph(row):
            """
            gets a row of the dataframe and appends a corresponding Data object into the scenegraph
            """
            nodeIndex = {}
            edgeAttributeIndex = {}
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

            sceneGraph.nodeIndex = nodeIndex
            sceneGraph.edgeAttributeIndex = edgeAttributeIndex
            
            # sceneGraphs.append(sceneGraph)

            row['SG'] = sceneGraph

            return row

        data = data.apply(getSceneGraph, axis=1)

        return data[['SG']]

    def readSceneGraphs(self, videoName : str) -> pd.DataFrame:
        """
        returns a dataframe of scene graphs for the video inside the dataset folder.
        """
        pathToFile = os.path.join(DATASET_NAME, videoName, SCENEGRAPH_FILE_NAME)
        data = pd.read_json(pathToFile, orient='index')

        def getLabels(row):
            """
            gets the label for each scene graph.
            """
            name = row.name

            row['id'] = name
            row['label'] = None

            return row

        return data

    def loadImageEmbeddings(self, videoName : str) -> pd.DataFrame:
        data = pd.read_json(os.path.join(DATASET_NAME, videoName, IMAGE_EMBEDDING_FILE_NAME), 
                            orient='index')
        
        def transform(row):
            row['imageEmbedding'] = torch.tensor(row[list(range(0, len(row)))], dtype=torch.float)

            return row

        data = data.apply(transform, axis=1)

        return data[['imageEmbedding']]

    def readPeripheralInputs(self, videoName : str):
        pass

    def loadAllData(self, videoName : str):
        sceneGraphs = self.loadSceneGraphData(videoName)
        imageEmbeddings = self.loadImageEmbeddings(videoName)
        peripheralInputs = self.readPeripheralInputs(videoName)

        def getLabels(row):
            """
            gets the label for each row.
            """
            imageName = row.name.split('/')[-1][:-4]
            label = imageName.split('_')[1]

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