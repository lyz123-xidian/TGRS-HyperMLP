import random

# 生成随机的节点和边
n = 128
G = {}
for i in range(n):
    G[i] = {'v': [random.randint(0, n - 1) for j in range(n)]}
    for j in range(n):
        G[i]['v'].append((i, j))

for i in range(n):
    for j in range(n):
        G[i]['weight'] = [random.randint(0, n - 1) for k in range(n)]

    # 生成 edge_index
num_nodes = 100
num_edges = 128
edge_index = []
for i in range(num_nodes):
    for j in range(num_edges):
        edge_index.append((i, j))

    # 打印 edge_index
for i in range(num_nodes):
    for j in range(num_edges):
        print(edge_index[i][j])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = x.view(-1, hidden_size)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



if __name__ == "__main__":
    img = torch.ones([12, 121, 128])
    # img2 = torch.randn([12, 1, 15, 15])
    model = GraphConvolution(in_features=128,out_features=6)
    out_img = model(img,)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
