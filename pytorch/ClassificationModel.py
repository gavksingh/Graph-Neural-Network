import os.path as osp
from collections import OrderedDict
import numpy as np
import torch
from torch import Tensor
import torch_geometric.transforms as transforms
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import Sequential, Linear
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
from sklearn.model_selection import StratifiedKFold, train_test_split
from CustomGTVConv import CustomGTVConv
from .CustomCheegerCutPool import CustomCheegerCutPool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################
# CONFIG
################################
message_passing_layers = 1
message_passing_channels = 32
message_passing_activation = "relu"
delta_coefficient = 2.0

mlp_hidden_layers = 1
mlp_hidden_channels = 32
mlp_activation = "relu"
total_variation_coeff = 0.5
balance_coefficient = 0.5

training_epochs = 100
batch_size = 16
learning_rate = 5e-4
l2_regularization_value = 0
patience_epochs = 10

results = {"accuracy_scores": []}

################################
# LOAD DATASET
################################
dataset_name = "PROTEINS"

data_path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_name)
dataset = TUDataset(data_path, "PROTEINS", use_node_attr=True, cleaned=True)

# Parameters
max_nodes = max(graph.num_nodes for graph in dataset)
num_output_classes = dataset.num_classes  # Dimension of the target

# Train/test split
indices = np.random.permutation(len(dataset))
split_validation, split_test = int(0.8 * len(dataset)), int(0.9 * len(dataset))
train_indices, validation_indices, test_indices = np.split(indices, [split_validation, split_test])
training_dataset = dataset[torch.tensor(train_indices).long()]
validation_dataset = dataset[torch.tensor(validation_indices).long()]
test_dataset = dataset[torch.tensor(test_indices).long()]
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

################################
# MODEL
################################
class ClassificationModel(torch.nn.Module):
    def __init__(self, num_output_classes, mp1, pool1, mp2, pool2, mp3):
        super().__init()

        self.mp1 = mp1
        self.pool1 = pool1
        self.mp2 = mp2
        self.pool2 = pool2
        self.mp3 = mp3
        self.output_layer = Linear(message_passing_channels, num_output_classes)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, batch: Tensor):
        
        # 1st block
        x = self.mp1(x, edge_index, edge_weight)
        x, mask = to_dense_batch(x, batch)
        adjacency = to_dense_adj(edge_index, edge_attr=edge_weight, batch=batch)
        x, adjacency, tv1, bal1 = self.pool1(x, adjacency, mask=mask)

        # 2nd block
        x = self.mp2(x, edge_index=adjacency, edge_weight=None)
        x, adjacency, tv2, bal2 = self.pool2(x, adjacency)

        # 3rd block
        x = self.mp3(x, edge_index=adjacency, edge_weight=None)
        x = x.mean(dim=1)  # Global mean pooling
        x = self.output_layer(x)

        return x, tv1 + tv2, bal1 + bal2

MP1 = [
    (CustomGTVConv(dataset.num_features if i==0 else message_passing_channels,
            message_passing_channels,
            act=message_passing_activation,
            delta_coeff=delta_coefficient),
        'x, edge_index, edge_weight -> x')
    for i in range(message_passing_layers)]

MP1 = Sequential('x, edge_index, edge_weight', MP1)

Pool1 = CustomCheegerCutPool(int(max_nodes//2),
                           mlp_channels=[message_passing_channels] + 
                                [mlp_hidden_channels for _ in range(mlp_hidden_layers)],
                           mlp_activation=mlp_activation,
                           totvar_coeff=total_variation_coeff,
                           balance_coeff=balance_coefficient,
                           return_selection=False,
                           return_pooled_graph=True)

MP2 = [
    (CustomGTVConv(message_passing_channels,
            message_passing_channels,
            act=message_passing_activation,
            delta_coeff=delta_coefficient),
        'x, edge_index, edge_weight -> x')
    for _ in range(message_passing_layers)]

MP2 = Sequential('x, edge_index, edge_weight', MP2)

Pool2 = CustomCheegerCutPool(int(max_nodes//4),
                           mlp_channels=[message_passing_channels] + 
                                [mlp_hidden_channels for _ in range(mlp_hidden_layers)],
                           mlp_activation=mlp_activation,
                           totvar_coeff=total_variation_coeff,
                           balance_coeff=balance_coefficient,
                           return_selection=False,
                           return_pooled_graph=True)

MP3 = [
    (CustomGTVConv(message_passing_channels,
            message_passing_channels,
            act=message_passing_activation,
            delta_coeff=delta_coefficient),
        'x, edge_index, edge_weight -> x')
    for _ in range(message_passing_layers)]

MP3 = Sequential('x, edge_index, edge_weight', MP3)

# Setup model
model = ClassificationModel(num_output_classes,
                            mp1=MP1,
                            pool1=Pool1,
                            mp2=MP2,
                            pool2=Pool2,
                            mp3=MP3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization_value)
loss_fn = torch.nn.CrossEntropyLoss()

        
################################
# TRAIN AND TEST
################################

def train():
    model.train()

    for data in training_loader:
        data.to(device)
        out, tv_loss, bal_loss = model(data.x, data.edge_index, data.edge_weight, data.batch)
        loss = tv_loss + bal_loss
        loss += loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

@torch.no_grad()       
def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data.to(device)
        out, tv_loss, bal_loss = model(data.x, data.edge_index, data.edge_weight, data.batch)
        loss = tv_loss + bal_loss + loss_fn(out, data.y)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return loss, correct / len(loader.dataset)

best_val_accuracy = 0
patience_count = patience_epochs
for epoch in range(1, training_epochs + 1):
    train()
    train_loss, train_accuracy = test(training_loader)
    val_loss, val_accuracy = test(validation_loader)
    test_loss, test_accuracy = test(test_loader)

    print(f"Epoch: {epoch:03d}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Test Acc: {test_accuracy: .4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        test_loss_at_best_val = test_loss
        test_accuracy_at_best_val = test_accuracy
        patience_count = patience_epochs
    else:
        patience_count -= 1
    if patience_count == 0:
        break

print("Test loss: {}. Test acc: {}".format(test_loss_at_best_val, test_accuracy_at_best_val))
