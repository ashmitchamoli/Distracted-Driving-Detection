import pandas as pd
from torch_geometric.data import Data
import torch
import os 

DATASET_NAME = 'Datasets/DDDatasetAnnotated'
SCENEGRAPH_FILE_NAME = 'sceneGraphs.json'

class Preprocessor:
    def __init__(self) -> None:
        pass

    def loadSceneGraphData(self, videoName : str) -> list[Data]:
        """
        dataPath : path to scene graph json file

        Returns a list of torch_geometric.data.Data objects.
        """
        data = self.readSceneGraphs(videoName)

        sceneGraps = []

        def getSceneGraph(row):
            """
            gets a row of the dataframe and appends a Data object into sceneGraps
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
            
            sceneGraps.append(sceneGraph)

        data.apply(getSceneGraph, axis=1)

        return sceneGraps

    def readSceneGraphs(self, videoName : str) -> pd.DataFrame:
        """
        returns a dataframe of scene graphs for the video inside the dataset folder.
        """
        pathToFile = os.path.join(DATASET_NAME, videoName, SCENEGRAPH_FILE_NAME)
        data = pd.read_json(pathToFile, orient='index')
        return data


