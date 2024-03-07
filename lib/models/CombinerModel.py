import torch
from lib.models.LinearClassifier import LinearClassifier
from lib.models.GNN import GNN

class CombinerModel(torch.nn.Module):
    def __init__(self, numSceneGraphFeatures : int, sceneGraphEmbeddingSize : int, imgEmbeddingSize : int, reducedImgEmbeddingSize : int, encoderHiddenLayers : list[int], numClasses : int, n_peripheralInputs : int, feedForwardHiddenLayers : list[int]) -> None:
        """
        Class for combining the GNN model with the Linearclassifier for training.
        """
        super().__init__()

        self.sceneGraphBlock = GNN(numSceneGraphFeatures, sceneGraphEmbeddingSize)
        self.ffnnClassifierBlock = LinearClassifier(imgEmbeddingSize, reducedImgEmbeddingSize, encoderHiddenLayers, sceneGraphEmbeddingSize, numClasses, n_peripheralInputs, feedForwardHiddenLayers)

    def forward(self, imgEmbedding, sceneGraphEmbedding, peripheralInputs):
        sceneGraphEmbedding = self.sceneGraphBlock(sceneGraphEmbedding)
        return self.ffnnClassifierBlock(imgEmbedding, sceneGraphEmbedding, peripheralInputs)

    def train(self, data):
        pass
