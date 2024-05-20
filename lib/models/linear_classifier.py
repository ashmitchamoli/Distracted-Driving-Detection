import torch

class LinearClassifier(torch.nn.Module):
    def __init__(self, imgEmbeddingSize : int, reducedImgEmbeddingSize : int, encoderHiddenLayers : list[int], sceneGraphEmbeddingSize : int, numClasses : int, n_peripheralInputs : int, feedForwardHiddenLayers : list[int]) -> None:
        """
        Args:
            imgEmbeddingSize (int): The size of the image embedding.
            reducedImgEmbeddingSize (int): The size of the reduced image embedding.
            encoderHiddenLayers (list[int]): List of hidden layer sizes for the image embedding encoder.
            sceneGraphEmbeddingSize (int): The size of the scene graph embedding.
            numClasses (int): The number of output classes.
            n_peripheralInputs (int): The number of peripheral inputs.
            feedForwardHiddenLayers (list[int]): List of hidden layer sizes for the feed forward network.
        
        Returns:
            None
        """
        super().__init__()

        self.imgEmbeddingSize = imgEmbeddingSize
        self.reducedImgEmbeddingSize = reducedImgEmbeddingSize
        self.encoderHiddenLayers = encoderHiddenLayers
        
        self.sceneGraphEmbeddingSize = sceneGraphEmbeddingSize
        self.n_peripheralInputs = n_peripheralInputs
        self.feedForwardHiddenLayers = feedForwardHiddenLayers

        self.numClasses = numClasses

        self.imgEmbeddingEncoder = torch.nn.Sequential()
        for i in range(len(self.encoderHiddenLayers)):
            linearLayer = torch.nn.Linear(self.imgEmbeddingSize if i == 0 else self.encoderHiddenLayers[i], 
                                          self.reducedImgEmbeddingSize if (i == len(self.encoderHiddenLayers) - 1) else self.encoderHiddenLayers[i+1])
            self.imgEmbeddingEncoder.append(linearLayer)
            if i == len(self.encoderHiddenLayers) - 1:
                self.imgEmbeddingEncoder.append(torch.nn.Tanh())
                continue # ! which activation function to add at the end?
            else:
                self.imgEmbeddingEncoder.append(torch.nn.Tanh())

        if len(self.encoderHiddenLayers) == 0:
            linearLayer = torch.nn.Linear(self.imgEmbeddingSize, self.reducedImgEmbeddingSize)
            self.imgEmbeddingEncoder.append(linearLayer)
            self.imgEmbeddingEncoder.append(torch.nn.Tanh())
            # ! which activation function to add at the end?

        # define linear layer here
        self.feedForwardLayer = torch.nn.Sequential()
        n_feedForwardInputs = reducedImgEmbeddingSize + sceneGraphEmbeddingSize + n_peripheralInputs
        for i in range(len(self.feedForwardHiddenLayers)):
            linearLayer = torch.nn.Linear(n_feedForwardInputs if i == 0 else self.feedForwardHiddenLayers[i],
                                          self.numClasses if i == len(self.feedForwardHiddenLayers) - 1 else self.feedForwardHiddenLayers[i+1])
            self.feedForwardLayer.append(linearLayer)
            if i == len(self.feedForwardHiddenLayers) - 1:
                # ! handle the softmax dim for batched input
                # self.feedForwardLayer.append(torch.nn.Softmax(dim=0))
                pass
            else:
                self.feedForwardLayer.append(torch.nn.Tanh())

        if len(self.feedForwardHiddenLayers) == 0:
            linearLayer = torch.nn.Linear(n_feedForwardInputs, self.numClasses)
            self.feedForwardLayer.append(linearLayer)
            # ! handle the softmax dim for batched input
            # self.feedForwardLayer.append(torch.nn.Softmax(dim=0))
        

    def forward(self, imgEmbedding, sceneGraphEmbedding, peripheralInputs):
        imgEmbedding = self.imgEmbeddingEncoder(imgEmbedding)
        # print("imgEmbedding:", imgEmbedding)

        # flatten scene graph embedding
        sceneGraphEmbedding = sceneGraphEmbedding.reshape(-1)

        # print(imgEmbedding.shape, sceneGraphEmbedding.shape, peripheralInputs.shape)

        # input = torch.cat((imgEmbedding.reshape(-1, imgEmbedding.shape[1]), 
        #                    sceneGraphEmbedding, 
        #                    peripheralInputs.reshape(-1, imgEmbedding.shape[1])), dim=1)
        
        # ! the following will work only when batch size is 1. change this later when you add batched input
        input = torch.cat((imgEmbedding, sceneGraphEmbedding, peripheralInputs))

        # input = torch.cat((imgEmbedding, sceneGraphEmbedding, peripheralInputs), dim=1)
        return self.feedForwardLayer(input)
