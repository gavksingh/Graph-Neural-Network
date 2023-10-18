from typing import List, Optional, Tuple, Union
import math
import torch
from torch import Tensor
from torch_geometric.nn.models.mlp import Linear
from torch_geometric.nn.resolver import custom_activation_resolver

class CustomCheegerCutPool(torch.nn.Module):
    r"""
    A custom implementation of the Cheeger cut pooling layer inspired by the paper
    `"Graph Neural Networks for Total Variation-Based Graph Clustering"
    <https://example.com/paper-link>`_.

    Args:
        k_clusters (int): 
            Number of clusters or output nodes 
        hidden_units (int, list of int): 
            Number of hidden units for each hidden layer in the MLP used to 
            compute cluster assignments. First integer must match the number 
            of input channels.
        activation (any): 
            Activation function between hidden layers of the MLP. 
            Must be compatible with `custom_activation_resolver`.
        include_selection (bool): 
            Whether to include the selection matrix. Set to False if 
            `include_pooled_graph` is True. (default: :obj:`False`)
        include_pooled_graph (bool): 
            Whether to include pooled node features and adjacency. Set to False if 
            `include_selection` is True. (default: :obj:`True`)
        use_bias (bool): 
            Whether to add a bias term to the MLP layers. (default: :obj:`True`)
        tv_loss_coeff (float): 
            Coefficient for the total variation graph loss component. (default: :obj:`1.0`)
        balance_loss_coeff (float): 
            Coefficient for the asymmetric norm loss component. (default: :obj:`1.0`)
    """

    def __init__(self, 
                 k_clusters: int, 
                 hidden_units: Union[int, List[int]], 
                 activation="relu",
                 include_selection: bool = False,
                 include_pooled_graph: bool = True,
                 use_bias: bool = True,
                 tv_loss_coeff: float = 1.0,
                 balance_loss_coeff: float = 1.0,
                 ):
        super().__init()

        if not include_selection and not include_pooled_graph:
            raise ValueError("Both include_selection and include_pooled_graph cannot be False")

        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]

        act = custom_activation_resolver(activation)
        in_channels = hidden_units[0]
        self.mlp = torch.nn.Sequential()
        for channels in hidden_units[1:]:
            self.mlp.append(Linear(in_channels, channels, bias=use_bias))
            in_channels = channels
            self.mlp.append(act)

        self.mlp.append(Linear(in_channels, k_clusters))
        self.k_clusters = k_clusters
        self.include_selection = include_selection
        self.include_pooled_graph = include_pooled_graph
        self.tv_loss_coeff = tv_loss_coeff
        self.balance_loss_coeff = balance_loss_coeff

        self.initialize_parameters()

    def initialize_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(
        self,
        node_features: Tensor,
        adjacency_matrix: Tensor,
        node_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Args:
            node_features (Tensor): 
                Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}` 
                with batch-size :math:`B`, (maximum) number of nodes :math:`N` for each graph,
                and feature dimension :math:`F`. The cluster assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}` is
                created within this method.
            adjacency_matrix (Tensor): 
                Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            node_mask (BoolTensor, optional): 
                Mask matrix :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` 
                indicating the valid nodes for each graph. (default: :obj:`None`)

        :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
            :class:`Tensor`, :class:`Tensor`, :class:`Tensor`)
        """
        node_features = node_features.unsqueeze(0) if node_features.dim() == 2 else node_features
        adjacency_matrix = adjacency_matrix.unsqueeze(0) if adjacency_matrix.dim() == 2 else adjacency_matrix

        cluster_assignments = self.mlp(node_features)
        cluster_assignments = torch.softmax(cluster_assignments, dim=-1)

        batch_size, n_nodes, _ = node_features.size()

        if node_mask is not None:
            node_mask = node_mask.view(batch_size, n_nodes, 1).to(node_features.dtype)
            node_features, cluster_assignments = node_features * node_mask, cluster_assignments * node_mask

        # Pooled features and adjacency
        if self.include_pooled_graph:
            pooled_node_features = torch.matmul(cluster_assignments.transpose(1, 2), node_features)
            pooled_adjacency_matrix = torch.matmul(torch.matmul(cluster_assignments.transpose(1, 2), adjacency_matrix), cluster_assignments) 

        # Total variation loss
        tv_loss = self.tv_loss_coeff * torch.mean(self.total_variation_loss(adjacency_matrix, cluster_assignments))
        
        # Balance loss
        balance_loss = self.balance_loss(cluster_assignments)

        if self.include_selection and self.include_pooled_graph:
            return cluster_assignments, pooled_node_features, pooled_adjacency_matrix, tv_loss, balance_loss
        elif self.include_selection and not self.include_pooled_graph:
            return cluster_assignments, tv_loss, balance_loss
        else:
            return pooled_node_features, pooled_adjacency_matrix, tv_loss, balance_loss

    def total_variation_loss(self, adjacency_matrix, cluster_assignments):
        batch_idx, node_i, node_j = torch.nonzero(adjacency_matrix, as_tuple=True)
        l1_norm = torch.sum(torch.abs(cluster_assignments[batch_idx, node_i, :] - cluster_assignments[batch_idx, node_j, :]), dim=(-1)) 
        loss = torch.sum(adjacency_matrix[batch_idx, node_i, node_j] * l1_norm)
        
        # Normalize the loss
        n_edges = len(node_i)
        loss *= 1 / (2 * n_edges)

        return loss

    def balance_loss(self, cluster_assignments):
        n_nodes = cluster_assignments.size()[-2]

        # k-quantile
        idx = int(math.floor(n_nodes / self.k_clusters))
        quant = torch.sort(cluster_assignments, dim=-2, descending=True)[0][:, idx, :] # shape [B, K]

        # Asymmetric l1-norm (continued)
        loss = (loss >= 0) * (self.k_clusters - 1) * loss + (loss < 0) * loss * -1 
        loss = torch.sum(loss, axis=(-1, -2)) # shape [B]
        loss = 1 / (n_nodes * (self.k_clusters - 1)) * (n_nodes * (self.k_clusters - 1) - loss)

        return loss


