import sys
sys.path.insert(1, '../utils')

import os.path as osp
import torch
from torch import Tensor
import torch_geometric.transforms as transforms
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric import utils
from torch_geometric.nn import Sequential
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from custom_metrics import custom_cluster_acc
from CustomGTVConv import CustomGTVConv
from CustomCheegerCutPool import CustomCheegerCutPool

torch.manual_seed(1) 
device = torch.device('cuda' if torch.cuda.is available() else 'cpu')

################################
# CONFIG
################################
dataset_id = "Cora"
message_passing_channels = 512
message_passing_layers = 2
message_passing_activation = "elu"
delta_coefficient = 0.311
mlp_hidden_channels = 256
mlp_hidden_layers = 1
mlp_activation = "relu"
total_variation_coeff = 0.785
balance_coefficient = 0.514
learning_rate = 1e-3
epochs = 500

################################
# LOAD DATASET
################################
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_id)
if dataset_id in ["Cora", "CiteSeer", "PubMed"]:
    dataset = Planetoid(path, dataset_id, transform=transforms.NormalizeFeatures())
elif dataset_id == "DBLP":
    dataset = CitationFull(path, dataset_id, transform=transforms.NormalizeFeatures())

data = dataset[0]
data = data.to(device)

############################################################################
# MODEL
############################################################################

class CustomNet(torch.nn.Module):

    def __init__(self):
        super().__init()

        # Message passing layers
        message_passing = [
            (CustomGTVConv(dataset.num_features if i == 0 else message_passing_channels,
                     message_passing_channels,
                     act=message_passing_activation,
                     delta_coeff=delta_coefficient),
             'x, edge_index, edge_weight -> x')
        for i in range(message_passing_layers)]
        
        self.message_passing = Sequential('x, edge_index, edge_weight', message_passing)
        
        # Pooling layer
        self.pool = CustomCheegerCutPool(
            dataset.num_classes,
            mlp_channels=[message_passing_channels] + [mlp_hidden_channels for _ in range(mlp_hidden_layers)],
            mlp_activation=mlp_activation,
            totvar_coeff=total_variation_coeff,
            balance_coeff=balance_coefficient,
            return_selection=True,
            return_pooled_graph=False)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor):

        # Propagate node features
        x = self.message_passing(x, edge_index, edge_weight)
        
        # Compute cluster assignment and obtain auxiliary losses
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
        s, tv_loss, bal_loss = self.pool(x, adj)

        return s.squeeze(0), tv_loss, bal_loss

model = CustomNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)

############################################################################
# TRAINING
############################################################################
def custom_train():
    model.train()
    optimizer.zero_grad()
    _, tv_loss, bal_loss = model(data.x, data.edge_index, data.edge_weight)
    loss = tv_loss + bal_loss
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def custom_test():
    model.eval()
    clust, _, _ = model(data.x, data.edge_index, data.edge_weight)
    return NMI(data.y.cpu(), clust.max(1)[1].cpu()), custom_cluster_acc(data.y.cpu().numpy(), clust.max(1)[1].cpu().numpy())[0]

patience = 50
best_loss = 1
nmi_at_best_loss = 0
acc_at_best_loss = 0
for epoch in range(1, epochs+1):
    train_loss = custom_train()
    nmi, acc = custom_test()
    print(f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, NMI: {nmi:.3f}, ACC: {acc*100: .3f}")
    if train_loss < best_loss:
        best_loss = train_loss
        nmi_at_best_loss = nmi
        acc_at_best_loss = acc
        patience = 50
    else:
        patience -= 1
    if patience == 0:
        break

print(f"NMI: {nmi_at_best_loss:.3f}, ACC: {acc_at_best_loss*100:.1f}")
