import torch
from torch.utils.data import Dataset
from typing import Literal

from lib.driving_dataset.Preprocessor import Preprocessor

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