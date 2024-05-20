import torch
import os
from sklearn.metrics import classification_report, accuracy_score
import tqdm
import pickle as pkl

from lib.models.linear_classifier import LinearClassifier
from lib.models.gnn import GNN
from lib.driving_dataset.DrivingDataset import DrivingDataset

class CombinerModel(torch.nn.Module):
    def __init__(self, numSceneGraphFeatures : int, sceneGraphEmbeddingSize : int, imgEmbeddingSize : int, reducedImgEmbeddingSize : int, encoderHiddenLayers : list[int], numClasses : int, n_peripheralInputs : int, feedForwardHiddenLayers : list[int]) -> None:
        """
        Class for combining the GNN model with the Linearclassifier for training.
        """
        super().__init__()

        self.sceneGraphBlock = GNN(numSceneGraphFeatures, sceneGraphEmbeddingSize)
        self.ffnnClassifierBlock = LinearClassifier(imgEmbeddingSize, reducedImgEmbeddingSize, encoderHiddenLayers, sceneGraphEmbeddingSize, numClasses, n_peripheralInputs, feedForwardHiddenLayers)

        self.parameterString = f"{numSceneGraphFeatures}-{sceneGraphEmbeddingSize}-{imgEmbeddingSize}-{reducedImgEmbeddingSize}-{encoderHiddenLayers}-{numClasses}-{n_peripheralInputs}-{feedForwardHiddenLayers}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.to(self.device)
        self.ffnnClassifierBlock.to(self.device)
        self.sceneGraphBlock.to(self.device)
        for epoch in range(epochs):
            runningTrainLoss = 0
            runningDevLoss = 0
            i = 0
            for item in tqdm.tqdm(dataLoader, desc="Training", leave=True):
                X, y = item[0] # ! using a batch size of 1 for now
                optimizer.zero_grad()

                # print(X[1].shape, X[2].shape)
                X[0] = X[0].to(self.device)
                X[1] = X[1].to(self.device)
                X[2] = X[2].to(self.device)
                output = self(X[0], X[1], X[2])

                y = y.to(self.device)
                loss = criterion(output, y)

                loss.backward()
                optimizer.step()

                runningTrainLoss += loss.item()

                if i < len(devDataset): # ! dependent on the fact that batch size is 1 for now
                    with torch.no_grad():
                        X_dev = devDataset.X[i]
                        X_dev[0] = X_dev[0].to(self.device)
                        X_dev[1] = X_dev[1].to(self.device)
                        X_dev[2] = X_dev[2].to(self.device)
                        output_dev = self(X_dev[0], X_dev[1], X_dev[2])
                        y_dev = devDataset.y[i].to(self.device)
                    runningDevLoss += criterion(output_dev, y_dev).item()
                    
                i+=1

            # self.trainLoss.append(runningTrainLoss / len(dataLoader))
            runningTrainLoss /= len(dataLoader)
            runningDevLoss /= len(devDataset.X)

            # metrics[epoch] = self.evaluate(drivingDataset, devDataset)
            metrics[epoch] = {'trainMetrics': {}}
            metrics[epoch]['devMetrics'] = self.evaluate(devDataset)
            metrics[epoch]['trainMetrics']['loss'] = runningTrainLoss
            metrics[epoch]['devMetrics']['loss'] = runningDevLoss
            
            print(f"Epoch {epoch+1} | Train loss: {runningTrainLoss:.6f} | Dev loss: {runningDevLoss:.6f} | Dev accuracy: {metrics[epoch]['devMetrics']['accuracy']:.3f}")

            self.metrics = metrics

            if epoch%5 == 0:
                self.saveModel(epoch)
                print(f"Model saved at epoch {epoch}")
                print("Computing train metrics...")
                metrics[epoch]['trainMetrics'] = self.evaluate(drivingDataset)

    def saveModel(self, epoch) -> None:
        path = f'./cache'
        if not os.path.exists(path):
            os.makedirs(path)
        
        modelDirName = f'CombinerModel_{self.parameterString}'
        if not os.path.exists(os.path.join(path, modelDirName)):
            os.makedirs(os.path.join(path, modelDirName))

        torch.save(self.state_dict(), os.path.join(path, modelDirName, f'model_{epoch}.pt'))

        pkl.dump(self.metrics, open(os.path.join(path, f'metrics_{epoch}.pkl'), 'wb'))

    def evaluate(self, dataset : DrivingDataset) -> dict:
        """
        returns a dictionary of metrics over the given dataset.
        """
        metricsDict = {}

        # iterate over the dataset to get predictions
        preds = []
        with torch.no_grad(): # ? a way to optimize this. way too expensive to iterate over each frame.
            for X in dataset.X:
                output = self(X[0], X[1], X[2])    
                preds.append(output.argmax().item())
        
        # iterate over the dev set to get predictions
        # devPreds = []
        # with torch.no_grad():
        #     for X in devDataset:
        #         output = self(X[0], X[1], X[2])    
        #         devPreds.append(output.argmax().item())

        preds = torch.tensor(preds)

        metricsDict['preds'] = preds
        
        # classification report
        metricsDict['report'] = classification_report(dataset.y, preds, zero_division=0)
        
        # accuracy score
        metricsDict['accuracy'] = accuracy_score(dataset.y, preds)

        return metricsDict
