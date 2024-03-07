import torch
from torch.nn import Sequential
from torch_geometric.nn import GCNConv, aggr

class GNN(torch.nn.Module):
    def __init__(self, numFeatures : int, embeddingSize : int = 2) -> None:
        super().__init__()
        self.convLayers = Sequential()
        self.conv1 = GCNConv(numFeatures, embeddingSize)
        self.convLayers.append(self.conv1)
        self.nodeAggregator = aggr.MeanAggregation()

    def forward(self, x, edge_index):
        h = None
        for conv in self.convLayers:
            h = conv(x, edge_index)
            h = h.tanh()

        # now using these node level features, compute a graph level embedding
        graphEmbedding = self.nodeAggregator(h, torch.arange(1))

        return graphEmbedding