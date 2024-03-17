import torch

from lib.models.LinearClassifier import LinearClassifier
from lib.models.GNN import GNN
from lib.driving_dataset.DrivingDataset import DrivingDataset

import tqdm

class CombinerModel(torch.nn.Module):
    def __init__(self, numSceneGraphFeatures : int, sceneGraphEmbeddingSize : int, imgEmbeddingSize : int, reducedImgEmbeddingSize : int, encoderHiddenLayers : list[int], numClasses : int, n_peripheralInputs : int, feedForwardHiddenLayers : list[int]) -> None:
        """
        Class for combining the GNN model with the Linearclassifier for training.
        """
        super().__init__()

        self.sceneGraphBlock = GNN(numSceneGraphFeatures, sceneGraphEmbeddingSize)
        self.ffnnClassifierBlock = LinearClassifier(imgEmbeddingSize, reducedImgEmbeddingSize, encoderHiddenLayers, sceneGraphEmbeddingSize, numClasses, n_peripheralInputs, feedForwardHiddenLayers)
    
    def forward(self, sceneGraph, imgEmbedding, peripheralInputs):
        sceneGraphEmbedding = self.sceneGraphBlock(sceneGraph.x, sceneGraph.edge_index)
        return self.ffnnClassifierBlock(imgEmbedding, sceneGraphEmbedding, peripheralInputs)

    def train(self, drivingDataset : DrivingDataset, devDataset : DrivingDataset, lr : float, epochs : int) -> None:
        dataLoader = torch.utils.data.DataLoader(drivingDataset, batch_size=1, shuffle=True, collate_fn=drivingDataset.customCollate)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        # metrics to store in each epoch
        metrics = {}

        # * for each epoch, store the following:
        # train loss, dev loss
        # train TP/FP/TN/FN, dev TP/FP/TN/FN - classwise
        # evaluation metrics such as precision/recall/f1 score on both train and test 
        # embeddings and output probability distribution for each sample

        for epoch in range(epochs):
            runningTrainLoss = 0
            runningDevLoss = 0
            i = 0
            for item in tqdm.tqdm(dataLoader, desc="Training", leave=True):
                X, y = item[0] # ! using a batch size of 1 for now
                optimizer.zero_grad()

                # print(X[1].shape, X[2].shape)
                output = self(X[0], X[1], X[2])

                loss = criterion(output, y)

                loss.backward()
                optimizer.step()

                runningTrainLoss += loss.item()

                if i < len(devDataset):
                    with torch.no_grad():
                        output = self(devDataset.X[i][0], devDataset.X[i][1], devDataset.X[i][2])
                    runningDevLoss += criterion(output, devDataset.y[i]).item()
                    
                i+=1

            # self.trainLoss.append(runningTrainLoss / len(dataLoader))

            print(f"Epoch {epoch+1} | Train loss: {runningTrainLoss:.3f} | Dev loss: {runningDevLoss:.3f}")

            metrics[epoch] = self.evaluate(drivingDataset, devDataset)
            metrics[epoch]['trainLoss'] = runningTrainLoss
            metrics[epoch]['devLoss'] = runningDevLoss

    def evaluate(self, trainDataset : DrivingDataset, devDataset : DrivingDataset) -> dict:
        metricsDict = {}

        trainPreds = []
        with torch.no_grad():
            for X in trainDataset:
                output = self(X[0], X[1], X[2])    
                trainPreds.append(output.argmax().item())
        
        devPreds = []
        with torch.no_grad():
            for X in devDataset:
                output = self(X[0], X[1], X[2])    
                devPreds.append(output.argmax().item())

        trainPreds = torch.tensor(trainPreds)
        devPreds = torch.tensor(devPreds)

        metricsDict['trainPreds'] = trainPreds
        metricsDict['devPreds'] = devPreds

        return metricsDict
