# Graph Machine Learning Notes

## Graph Theory Basics
**Graph** $G$ &mdash; a pair $(V, E)$, where $V$ is a set of nodes, $E$ is a set of edges. <br>
Let $A$ &mdash; graph adjacency matrix. <br>
Let $D$ &mdash; graph degree matrix, that means $D = (\deg(v_1), \ldots, \deg(v_n))$. <br>
**Graph Laplacian matrix** is defined by the formula
$$L = D - A.$$

## Message-Passing Neural Networks (MPNNs)
A [paper](https://arxiv.org/pdf/1704.01212) with a good review of several MPNN architectures.

### Graph Convolutional Network (GCN)
[Original paper](https://arxiv.org/abs/1609.02907)

### Graph Isomorphism Network (GIN)
[Original paper](https://arxiv.org/abs/1810.00826)

### Graph Attention Network (GAT)
[Original paper](https://arxiv.org/abs/1710.10903)

### GATv2
[Original paper](https://arxiv.org/abs/2105.14491)

## Graph Transformer Network
[Original paper](https://arxiv.org/abs/1911.06455)

### Graphormer
[Original paper](https://arxiv.org/abs/2106.05234)

### Structure-Aware Transformer (SAT)
[Original paper](https://arxiv.org/pdf/2202.03036)

### Spectral Attention Network (SAN)
[Original paper](https://arxiv.org/abs/2106.03893)


## Generative Models on Graphs
[Good survey](https://arxiv.org/abs/2302.02591)

### Generative Diffusion Models on Graphs
#### EDP-GNN
[Original paper](https://arxiv.org/abs/2003.00638) <br>
It is the very first score matching based diffusion method for undirected graph generation.

This model can generate only adjacency matrices, not attributes.

#### DiGress
[Original paper](https://arxiv.org/abs/2209.14734)
This method extends the DDPM algorithm to
generate graphs with categorical node and edge attributes.
####
#### DiPhon???