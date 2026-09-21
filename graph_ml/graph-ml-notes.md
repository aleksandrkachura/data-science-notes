# Graph Machine Learning Notes

# Basic graph notation

A graph is commonly written as

$G = (V, E)$,

where:

- $V$ is the set of nodes, also called vertices.
- $E$ is the set of edges, also called links.

Let

$|V| = n$

be the number of nodes and

$|E| = m$

be the number of edges.

A node $v \in V$ represents an entity, object, or event.

An edge $e \in E$ represents a relationship, interaction, or connection between nodes.

For an edge connecting nodes $u$ and $v$, we may write

$e = (u,v).$

Depending on the graph type, $(u,v)$ may be equivalent to $(v,u)$, or the two may represent different relationships.

---

## 3.1 Topology

The topology of a graph refers to its structural organization:

- Which nodes are connected?
- How many neighbors does each node have?
- Are there densely connected regions?
- Are there hubs?
- Are there isolated nodes?
- Are there short paths between nodes?

Topology is distinct from node and edge attributes.

For example, in a social network:

- Topology tells us who is connected to whom.
- Node attributes may include age, location, or interests.
- Edge attributes may include interaction frequency or relationship strength.

---

## 3.2 Neighborhoods

For a node \(v\), its neighborhood is the set of nodes connected to it.

For an undirected graph:

\[
\mathcal{N}(v) = \{u \in V : (u,v) \in E\}.
\]

For a directed graph, it is useful to distinguish:

- In-neighbors: nodes with edges pointing toward \(v\).
- Out-neighbors: nodes to which \(v\) has outgoing edges.

\[
\mathcal{N}_{\text{in}}(v)
=
\{u : (u,v) \in E\},
\]

\[
\mathcal{N}_{\text{out}}(v)
=
\{u : (v,u) \in E\}.
\]

Graph neural networks often update a node representation by aggregating information from one or both of these neighborhoods.

## Graph Types in Machine Learning
1. Directed or undirected.
2. Weighted or unweighted.
3. With or without node attributes.
4. With or without edge attributes.
5. Homogeneous or heterogeneous.
6. Static or dynamic.
7. Connected or disconnected.
8. Simple graph or multigraph.
9. Pairwise graph or hypergraph.
10. One graph or a collection of graphs.
11. Spatial or non-spatial.
12. Sparse or dense.
### Directedness
- Undirected
- Directed

### Weights
- Unweighted
- Weighted

# 4. Directed and undirected graphs

## 4.1 Undirected graphs

In an undirected graph, an edge represents a symmetric relationship:

\[
(u,v) \in E
\iff
(v,u) \in E.
\]

The edge between \(u\) and \(v\) has no direction.

### Examples

- Friendship networks
- Collaboration networks
- Physical connections
- Road networks when direction is ignored
- Protein interaction networks
- Similarity graphs

### Example

```text
A ----- B
|       |
|       |
C ----- D
```

The connection between \(A\) and \(B\) means the same thing as the connection between \(B\) and \(A\).

---

## 4.2 Directed graphs

In a directed graph, an edge has a source and a destination:

\[
(u,v) \in E
\]

means that the edge points from \(u\) to \(v\).

In general,

\[
(u,v) \neq (v,u).
\]

### Examples

- Citation networks
- Web hyperlinks
- Follower networks
- Financial transactions
- Communication networks
- Transportation networks with one-way roads
- Information diffusion networks

### Example

```text
A -----> B
^        |
|        v
D <----- C
```

The edge \(A \to B\) is different from \(B \to A\).

---

## 4.3 Why direction matters in graph ML

Direction often carries important semantic information.

In a citation graph:

- \(A \to B\) may mean “paper \(A\) cites paper \(B\).”

In a transaction graph:

- \(A \to B\) may mean “money was transferred from \(A\) to \(B\).”

Ignoring direction can remove useful information. Some methods handle direction by:

- Using separate incoming and outgoing aggregations.
- Treating each directed edge as two distinct edge types.
- Converting the graph to an undirected graph.
- Using directed message-passing operators.

---

# 5. Weighted and unweighted graphs

## 5.1 Unweighted graphs

In an unweighted graph, an edge only indicates whether a connection exists.

The graph answers a yes-or-no question:

\[
(u,v) \in E?
\]

Examples:

- Two users are connected.
- Two web pages are linked.
- Two proteins interact.

---

## 5.2 Weighted graphs

In a weighted graph, each edge has an associated value:

\[
w : E \rightarrow \mathbb{R}.
\]

The value \(w_{uv}\) may represent:

- Strength of interaction
- Distance
- Cost
- Probability
- Frequency
- Capacity
- Similarity
- Travel time

### Examples

- Road networks: travel time or distance.
- Social networks: number of interactions.
- Communication networks: bandwidth.
- Similarity graphs: similarity score.
- Financial networks: transaction amount.

A weighted edge can be represented as

\[
e_{uv} = (u,v,w_{uv}).
\]

---

## 5.3 Positive and signed weights

Most weighted graphs use nonnegative weights:

\[
w_{uv} \geq 0.
\]

However, some graphs contain signed edges:

\[
w_{uv} \in \mathbb{R}.
\]

Examples:

- Trust and distrust networks
- Friend and enemy relationships
- Positive and negative correlations
- Signed social interactions

Signed graphs often require specialized models because positive and negative relationships may behave differently.

---

# 6. Graphs with node attributes

A node-attributed graph associates a feature vector with each node.

Let

\[
\mathbf{x}_v \in \mathbb{R}^{d}
\]

denote the feature vector of node \(v\).

The node feature matrix is

\[
X \in \mathbb{R}^{n \times d},
\]

where each row corresponds to one node.

### Examples of node attributes

In a social network:

- Age
- Location
- Profession
- Interests
- Account activity

In a molecular graph:

- Atom type
- Formal charge
- Hybridization
- Aromaticity

In a citation network:

- Word counts
- Text embeddings
- Publication year
- Author information

In a traffic network:

- Current speed
- Traffic volume
- Weather conditions
- Road capacity

---

## 6.1 Graphs without node attributes

Some graphs contain only topology.

In this case, the input may consist only of:

\[
G = (V,E).
\]

A model may use:

- One-hot node identifiers.
- Degree features.
- Random initial features.
- Positional encodings.
- Structural features such as centrality.
- Learned node embeddings.

A graph with no explicit node features is sometimes called a structure-only graph.

---

## 6.2 Node labels versus node attributes

These concepts should be distinguished.

### Node attributes

Attributes are input information available to the model:

\[
\mathbf{x}_v.
\]

### Node labels

Labels are usually prediction targets:

\[
y_v.
\]

For example, in a citation network:

- Node attribute: the text of a paper.
- Node label: the paper's research area.

A node can have both attributes and labels.

---

# 7. Graphs with edge attributes

An edge may also have a feature vector:

\[
\mathbf{e}_{uv} \in \mathbb{R}^{r}.
\]

The edge feature tensor or matrix stores information about relationships.

### Examples

In a transportation graph:

- Distance
- Speed limit
- Number of lanes
- Road type

In a molecular graph:

- Bond type
- Bond order
- Aromaticity
- Ring membership

In a transaction graph:

- Transaction amount
- Currency
- Timestamp
- Transaction type

In a communication graph:

- Message type
- Duration
- Frequency
- Channel

---

## 7.1 Edge labels and edge attributes

As with nodes, edge attributes are often inputs, while edge labels are prediction targets.

For example:

- Edge attribute: transaction amount.
- Edge label: fraudulent or legitimate.

Or:

- Edge attribute: chemical bond type.
- Edge label: whether a new bond should exist.

---

## 7.2 Graphs without edge attributes

Some graphs only indicate connectivity.

For example:

\[
e_{uv} = 1
\]

may simply mean that nodes \(u\) and \(v\) are connected.

In such graphs, the main source of information is the topology.

---

# 8. Homogeneous and heterogeneous graphs

## 8.1 Homogeneous graphs

A homogeneous graph contains one node type and one edge type.

\[
G = (V,E)
\]

All nodes have the same general interpretation, and all edges represent the same kind of relationship.

### Examples

- A friendship graph containing only users and friendship edges.
- A citation graph containing only papers and citation edges.
- A protein interaction graph containing only proteins and interaction edges.

A homogeneous graph may still contain node or edge attributes.

---

## 8.2 Heterogeneous graphs

A heterogeneous graph contains multiple node types, edge types, or both.

A common representation is

\[
G = (V,E,\tau,\phi),
\]

where:

- \(\tau(v)\) gives the type of node \(v\).
- \(\phi(e)\) gives the type of edge \(e\).

### Examples

A bibliographic graph may contain:

- Authors
- Papers
- Venues
- Topics

And relationships such as:

- Author writes paper.
- Paper cites paper.
- Paper published in venue.
- Paper discusses topic.

A recommendation graph may contain:

- Users
- Products
- Categories

And relationships such as:

- User rates product.
- User purchases product.
- Product belongs to category.

---

## 8.3 Why heterogeneity matters

Different edge types often have different meanings.

For example:

- “User purchased product”
- “User viewed product”
- “Product belongs to category”

These relationships should not necessarily be treated as interchangeable.

Heterogeneous graph models may use:

- Relation-specific transformations.
- Type-specific parameters.
- Meta-paths.
- Separate message-passing functions for each relation type.

---

# 9. Static and dynamic graphs

## 9.1 Static graphs

A static graph is assumed not to change during the period of analysis.

\[
G = (V,E)
\]

is fixed.

Node and edge features may still exist, but the topology is treated as constant.

### Examples

- A fixed molecular graph.
- A completed citation network snapshot.
- A road map used for structural analysis.
- A static knowledge graph.

Static does not necessarily mean that the real-world system never changes. It may simply mean that we analyze one snapshot.

---

## 9.2 Dynamic graphs

A dynamic graph changes over time.

A dynamic graph may be represented as a sequence:

\[
G_1, G_2, \ldots, G_T,
\]

where \(G_t\) is the graph at time \(t\).

Changes may include:

- New nodes appearing.
- Nodes disappearing.
- New edges forming.
- Edges disappearing.
- Edge weights changing.
- Node attributes changing.
- Edge attributes changing.

---

## 9.3 Discrete-time dynamic graphs

In a discrete-time representation, the graph is observed at distinct time steps:

\[
G_t = (V_t,E_t,X_t).
\]

For example, a social network may be recorded once per day.

```text
Day 1:  A --- B

Day 2:  A --- B --- C

Day 3:  A --- B --- C
         \         /
          ---------
```

---

## 9.4 Continuous-time or event-based graphs

Some graph data are better represented as a sequence of events:

\[
(u_i,v_i,t_i,\mathbf{e}_i),
\]

where:

- \(u_i\) is the source node.
- \(v_i\) is the destination node.
- \(t_i\) is the event time.
- \(\mathbf{e}_i\) contains optional event features.

Examples:

- A financial transaction occurs at a particular time.
- A user sends a message.
- A sensor reports a measurement.
- A user interacts with an item.

Event-based representations preserve more precise temporal information than regularly sampled snapshots.

---

## 9.5 Why time matters

The same graph structure can have different meanings depending on time.

For example:

- A user's interests may change.
- Fraudulent behavior may emerge as a temporal pattern.
- Traffic congestion depends on time of day.
- A citation network grows as new papers are published.
- Social interactions may become more predictive when recent interactions are weighted more heavily.

Dynamic graph learning may need to model:

- Temporal ordering.
- Recency.
- Periodicity.
- Delayed effects.
- Structural changes.
- Node birth and disappearance.

---

# 10. Single graphs and graph collections

## 10.1 A single large graph

In some tasks, the data consist of one large graph.

Examples:

- One social network.
- One citation network.
- One transportation network.
- One knowledge graph.

The learning task may involve:

- Node classification.
- Link prediction.
- Community detection.
- Anomaly detection.
- Graph-level statistics.

---

## 10.2 A collection of graphs

In other tasks, each example is an individual graph:

\[
\mathcal{D} = \{G_1,G_2,\ldots,G_N\}.
\]

The graphs may have:

- Different numbers of nodes.
- Different numbers of edges.
- Different structures.
- Different node and edge features.

### Examples

- Molecule property prediction.
- Protein classification.
- Program analysis.
- 3D object recognition.
- Classifying brain connectivity networks.

The learning task often involves predicting one label per graph:

\[
\hat{y}_i = f_\theta(G_i).
\]

---

## 10.3 Why this distinction matters

A model designed for one large graph may not directly solve a graph classification problem over many graphs.

The main distinction is the prediction level:

| Prediction level | Typical example |
|---|---|
| Node-level | Classify users or papers |
| Edge-level | Predict future interactions |
| Graph-level | Predict molecular toxicity |
| Subgraph-level | Classify a functional motif |

---

# 11. Connected and disconnected graphs

## 11.1 Connected graphs

An undirected graph is connected if there is a path between every pair of nodes.

For every \(u,v \in V\), there exists a path from \(u\) to \(v\).

Examples:

- A connected road network.
- A connected molecule.
- A fully integrated communication network.

---

## 11.2 Disconnected graphs

A graph is disconnected if it contains multiple connected components.

```text
A --- B       E --- F
      |             |
      C             G
```

This graph has two connected components.

Disconnected graphs arise naturally in:

- User interaction data.
- Scientific networks.
- Knowledge graphs.
- Collections of independent entities.
- Graphs after filtering or sampling.

---

## 11.3 Why connected components matter

Disconnected components can affect:

- Message passing.
- Information flow.
- Diffusion processes.
- Spectral methods.
- Graph-level pooling.
- The interpretation of node embeddings.

A node in one component cannot receive information from a node in another component unless additional connections or global mechanisms are introduced.

---

# 12. Simple graphs, multigraphs, and hypergraphs

## 12.1 Simple graphs

A simple graph usually has:

- No self-loops.
- At most one edge between two nodes.
- No multiple edges representing different relationships.

This is a convenient mathematical abstraction, but many real datasets are more complex.

---

## 12.2 Self-loops

A self-loop is an edge from a node to itself:

\[
(v,v) \in E.
\]

Self-loops may represent:

- An entity interacting with itself.
- Persistence of a node's own information.
- A computational mechanism in graph neural networks.

Many GNN layers add self-loops so that a node can retain its own representation during message passing.

---

## 12.3 Multigraphs

A multigraph may contain multiple edges between the same pair of nodes.

For example, two users may:

- Send messages.
- Make phone calls.
- Exchange files.
- Follow each other.

These can be represented as separate edges, possibly with different types or timestamps.

---

## 12.4 Hypergraphs

A standard graph edge connects two nodes. A hyperedge can connect any number of nodes:

\[
e \subseteq V.
\]

Example:

\[
e = \{v_1,v_2,v_3,v_4\}.
\]

### Applications

- Group conversations.
- Co-authorship.
- Set-based interactions.
- Course enrollment.
- Package contents.
- Multi-person meetings.

A hypergraph cannot always be represented accurately by ordinary pairwise edges without losing information.

---

# 13. Bipartite and multipartite graphs

## 13.1 Bipartite graphs

A bipartite graph has two disjoint node sets:

\[
V = U \cup W,
\]

such that edges connect nodes from different sets:

\[
E \subseteq U \times W.
\]

There are no edges between two nodes both in \(U\), or both in \(W\).

### Examples

- Users and products.
- Students and courses.
- Authors and papers.
- Customers and businesses.
- Jobs and applicants.

A recommendation system is often naturally represented as:

```text
Users              Items

User 1 ----------- Item A
User 1 ----------- Item C
User 2 ----------- Item B
User 3 ----------- Item A
```

---

## 13.2 Multipartite graphs

A multipartite graph contains more than two node partitions.

For example, a scholarly graph may include:

- Authors
- Papers
- Conferences
- Institutions

Different types of edges connect different partitions.

Multipartite graphs are a special case of heterogeneous graphs.

---

# 14. Knowledge graphs and relational graphs

A knowledge graph represents entities and typed relations.

A common representation is a set of triples:

\[
(s,r,o),
\]

where:

- \(s\) is the subject.
- \(r\) is the relation.
- \(o\) is the object.

Example:

\[
(\text{Paris}, \text{capital\_of}, \text{France}).
\]

Knowledge graphs are usually:

- Directed.
- Typed.
- Heterogeneous.
- Often incomplete.
- Sometimes temporal.

### Common tasks

- Link prediction.
- Relation prediction.
- Entity classification.
- Knowledge graph completion.
- Question answering.

A key distinction from ordinary graphs is that the relation type is often central to the meaning of an edge.

---

# 15. Spatial graphs and geometric graphs

Some graphs are embedded in physical or geometric space.

Each node may have a coordinate:

\[
\mathbf{p}_v \in \mathbb{R}^{d}.
\]

Edges may be based on:

- Physical connections.
- Distance thresholds.
- Nearest neighbors.
- Visibility.
- Geometric constraints.

### Examples

- Sensor networks.
- Molecular structures.
- 3D point clouds.
- Road networks.
- Meshes.
- Weather stations.

Geometric graphs may use both:

- Graph connectivity.
- Continuous spatial information.

This motivates geometric deep learning, which aims to respect symmetries such as translation, rotation, and permutation.

---

# 16. Common graph representations

## 16.1 Adjacency matrix

For a graph with $n$ nodes, the adjacency matrix is

$$
A \in \mathbb{R}^{n \times n}.
$$

For an unweighted graph:

$$A_{uv} =
\begin{cases}
1 & \text{if } (u,v) \in E,\\
0 & \text{otherwise.}
\end{cases}$$

For a weighted graph, $A_{uv}$ may equal the edge weight $w_{uv}$.

For an undirected graph: $$A = A^\top.$$

For a directed graph, $A$ is not necessarily symmetric.

---

## 16.2 Degree matrix

For an undirected graph, the degree of node $v$ is

$$
d_v = \sum_u A_{vu}.
$$

The degree matrix is diagonal:

$$
D_{vv}=d_v.
$$

That is,

$$
D =
\begin{bmatrix}
d_1 & 0 & \cdots & 0\\
0 & d_2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & d_n
\end{bmatrix}.
$$

For directed graphs, we distinguish:

- In-degree.
- Out-degree.

---

## 16.3 Node feature matrix

If each node has a $$d$$-dimensional feature vector,

$$
X =
\begin{bmatrix}
\mathbf{x}_1^\top\\
\mathbf{x}_2^\top\\
\vdots\\
\mathbf{x}_n^\top
\end{bmatrix}
\in \mathbb{R}^{n \times d}.
$$

The graph input may therefore be represented by

$$
(A,X).
$$

---

## 16.4 Edge list

Instead of storing a dense adjacency matrix, we may store the edges as a list:

```text
source: [0, 1, 2, 2]
target: [1, 2, 0, 3]
```

An edge list is more efficient for sparse graphs.

Optional edge attributes can be stored alongside it:

```text
edge_features:
[
  [...],
  [...],
  [...],
  [...]
]
```

---

## 16.5 Incidence matrix

An incidence matrix describes which nodes belong to which edges.

For an undirected graph with $$n$$ nodes and $$m$$ edges:

$$
B \in \mathbb{R}^{n \times m}.
$$

Each column corresponds to one edge.

For directed graphs, the incidence matrix can encode source and destination using different signs.

---

# 17. Graph sparsity

Most real-world graphs are sparse.

A graph is sparse when

$$
m \ll n^2.
$$

A dense adjacency matrix requires $O(n^2)$ storage, which can be impractical for large graphs.

Sparse graph algorithms instead work with the edge list and typically require storage proportional to

$$
O(n+m).
$$

Examples of sparse graphs:

- Social networks.
- Citation networks.
- Road networks.
- Web graphs.
- Transaction networks.

Some graphs are comparatively dense:

- Small molecular graphs.
- Fully connected similarity graphs.
- Correlation networks.
- Attention graphs constructed by connecting every pair of nodes.

---

# 18. Important graph properties

When describing a graph, it is useful to ask the following questions:

| Property | Possible values |
|---|---|
| Direction | Directed, undirected, mixed |
| Weights | Weighted, unweighted, signed |
| Node features | Present, absent |
| Edge features | Present, absent |
| Node types | Homogeneous, heterogeneous |
| Edge types | Single type, multiple types |
| Time | Static, dynamic, temporal |
| Size | Small, medium, large-scale |
| Connectivity | Connected, disconnected |
| Multiplicity | Simple graph, multigraph |
| Higher-order relations | Pairwise graph, hypergraph |
| Geometry | Non-geometric, spatial, geometric |
| Dataset structure | One graph, collection of graphs |

A single graph can have many properties simultaneously.

For example, a transaction network might be:

- Directed.
- Weighted.
- Dynamic.
- Heterogeneous.
- Edge-attributed.
- Disconnected.
- A multigraph.
- Sparse.

---

# 19. A graph taxonomy by machine learning task

## 19.1 Node-level tasks

The goal is to predict something about each node.

Examples:

- Classify a paper by topic.
- Predict whether a user is fraudulent.
- Estimate a sensor's future value.
- Predict a protein's function.

Typical output:

$$
\hat{y}_v = f_\theta(G,v).
$$

---

## 19.2 Edge-level tasks

The goal is to predict something about an edge or node pair.

Examples:

- Predict whether two users will connect.
- Predict a missing citation.
- Estimate transaction risk.
- Predict the type of relation between two entities.

Typical output:

$$
\hat{y}_{uv} = f_\theta(G,u,v).
$$

---

## 19.3 Graph-level tasks

The goal is to predict a property of an entire graph.

Examples:

- Predict molecular toxicity.
- Classify a protein.
- Predict the behavior of a physical system.
- Classify a brain network.

Typical output:

$$
\hat{y}_G = f_\theta(G).
$$

---

## 19.4 Subgraph-level tasks

The goal is to predict a property of a region or substructure.

Examples:

- Detect a fraudulent transaction pattern.
- Identify a functional molecular motif.
- Find a community.
- Classify a local structure.

---

# 20. Why permutation invariance is important

The ordering of nodes in a graph is arbitrary.

If we permute the rows and columns of the adjacency matrix and reorder the corresponding node features, the underlying graph has not changed.

Let $P$ be a permutation matrix. Then a relabeled graph can be represented as

$$
A' = P A P^\top,
$$

$$
X' = P X.
$$

A graph-level prediction should remain unchanged:

$$
f(A,X) = f(PAP^\top, PX).
$$

This property is called permutation invariance.

For node-level predictions, the outputs should transform consistently with the node relabeling. This is called permutation equivariance.

Graph neural networks are generally designed to satisfy these properties.

---

# 21. How graph type affects graph neural networks

A typical message-passing layer has the form

$$
\mathbf{h}_v^{(k+1)}
=
\operatorname{UPDATE}^{(k)}
\left(
\mathbf{h}_v^{(k)},
\operatorname{AGGREGATE}^{(k)}
\left(
\left\{
\mathbf{h}_u^{(k)} : u \in \mathcal{N}(v)
\right\}
\right)
\right).
$$

The graph type influences this equation.

### Undirected graph

Use a neighborhood without direction:

$$
\mathcal{N}(v).
$$

### Directed graph

Use separate incoming and outgoing neighborhoods:

$$
\mathcal{N}_{\text{in}}(v),
\qquad
\mathcal{N}_{\text{out}}(v).
$$

### Edge-attributed graph

Include edge features:

$$
\mathbf{h}_v^{(k+1)}
=
\operatorname{UPDATE}
\left(
\mathbf{h}_v^{(k)},
\sum_{u \in \mathcal{N}(v)}
\psi
\left(
\mathbf{h}_u^{(k)},
\mathbf{e}_{uv}
\right)
\right).
$$

### Heterogeneous graph

Use relation-specific message functions:

$$
\mathbf{m}_v
=
\sum_{r \in \mathcal{R}}
\sum_{u \in \mathcal{N}_r(v)}
\psi_r(\mathbf{h}_u).
$$

### Dynamic graph

Include time:

$$
\mathbf{h}_v(t)
=
f_\theta
\left(
G_{\leq t},
v,t
\right).
$$

Thus, graph classification is not only terminology. It determines how information should be represented and propagated.

---

# 22. Worked example: classifying a graph

Consider a network of users and products.

- Users can purchase products.
- Products belong to categories.
- Purchases have timestamps and prices.
- New purchases arrive over time.
- A user may purchase the same product multiple times.

This graph can be classified as:

| Property | Classification |
|---|---|
| Direction | Directed |
| Node types | Heterogeneous |
| Edge types | Multiple |
| Edge attributes | Present |
| Weights | Weighted, if purchase amounts are used |
| Time | Dynamic |
| Multiplicity | Multigraph, if repeated purchases are retained |
| Connectivity | Potentially disconnected |
| Dataset structure | One large graph or a collection of user-centered subgraphs |

Possible machine learning tasks include:

- Predict whether a user will purchase a product.
- Predict the next product a user will purchase.
- Detect fraudulent transactions.
- Classify users.
- Recommend products.
- Predict product categories.

---

### Summary

A graph can be described along many independent dimensions:

1. Directed or undirected.
2. Weighted or unweighted.
3. With or without node attributes.
4. With or without edge attributes.
5. Homogeneous or heterogeneous.
6. Static or dynamic.
7. Connected or disconnected.
8. Simple graph or multigraph.
9. Pairwise graph or hypergraph.
10. One graph or a collection of graphs.
11. Spatial or non-spatial.
12. Sparse or dense.

These properties affect:

- Graph representation.
- Neighborhood definition.
- Message passing.
- Model architecture.
- Training strategy.
- Evaluation methodology.
- The meaning of predictions.


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
### Problem statement
**Given:**
- a set of graphs $\{G_i\}$ sampled from the data distribution $p_{\text{data}}(G)$.
- the model distribution $p_{\text{model}}(G; \theta)$ with parameters $\theta$.

**Goals:**
- *Density estimation:* to learn the distribution $p_{\text{model}}(G; \theta)$.
- *Sampling:* to generate new graphs from $p_{\text{model}}(G; \theta)$.

### Generative Model Basics
### Deep Generative Models
### GraphRNN
[Original paper](https://arxiv.org/abs/1802.08773)

**Autoregressive approach:** $p_{\text{model}}(G; \theta)$ is used for both density estimation and sampling. \
**Idea &mdash; chain rule:**
$$p_{\text{model}}(G; \theta) = \prod \limits_{t = 1}^n p_{\text{model}}(G_t | G_1, \ldots, G_{t - 1}; \theta).$$

**GraphRNN idea:** generating graphs via sequentially adding nodes and edges.

Graph $G$ with node ordering $\pi$ can be uniquely mapped into a sequence of node and edge additions $S^{\pi}$.

The sequence $S^{\pi}$ has 2 levels
($S$ is a sequence of sequences):
- Node-level: add nodes, one at a time.
- Edge-level: add edges between existing nodes.

**Node-level:** at each step, a new node is added.

Each node-level step is an edge-level sequence. \
**Edge-level:** at each step, add a new edge.

GraphRNN has a node-level RNN and an
edge-level RNN. \
Relationship between the two RNNs:
- Node-level RNN generates the initial state for edge-level RNN.
- Edge-level RNN sequentially predict if the new node will connect to each of the previous node.

**FROM THE ORIGINAL PAPER**

Graph is represented in the form of adjacency matrix. Denote by $\pi$ the node ordering (the permutation function) that maps nodes to rows/columns of an adjacency matrix and by $\Pi$ &mdash; the set of all $n!$ possible node permutations. Under a node ordering $\pi$, a graph $G$ can then be represented by the adjacency matrix $A^{\pi}\in\mathbb{R}^{n\times n}$, where $A^\pi_{i,j} = \mathbb{I}[(\pi(v_i),\pi(v_j))\in E]$. Note that elements in the set of adjacency matrices $A^\Pi=\{A^\pi | \pi \in \Pi\}$ all correspond to the same underlying graph.

The goal of *learning generative models of graphs* is to learn a distribution $p_{model}(G)$ over graphs, based on a set of observed graphs $\mathbb{G}=\{G_1,...,G_s\}$ sampled from data distribution $p(G)$, where each graph $G_i$ may have a different number of nodes and edges. 
When representing $G\in\mathbb{G}$, we further assume that we may observe any node ordering $\pi$ with equal probability, i.e.,
$p(\pi)=\frac{1}{n!}, \forall \pi \in \Pi$.
Thus, the generative model needs to be capable of generating graphs where each graph could have exponentially many representations, which is distinct from generative models for images, text, and time series.

#### Modeling graphs as sequences
Consider a mapping $f_S$ from graphs to sequences, where 
for a graph $G \sim p(G)$ with $n$ nodes under node ordering $\pi$, we have
$$
  S^\pi=f_S(G,\pi)=(S^\pi_1, \ldots ,S^\pi_{n}),
$$
where each element $S^\pi_i\in \{0,1\}^{i-1}, i\in \{1,...,n\}$ is an adjacency vector representing the edges between node $\pi(v_{i})$ and the previous nodes $\pi(v_{j}), j\in \{1,...,i-1\}$ already in the graph (we prohibit self-loops and  $S^\pi_1$ is defined as an empty vector):
For undirected graphs, $S^\pi$ determines a unique graph $G$, and we write the mapping as $f_G(\cdot)$ where $f_G(S^{\pi})=G$. 

Thus, instead of learning $p(G)$, whose sample space cannot be easily characterized, 
we sample the auxiliary $\pi$ to get the observations of $S^\pi$ and learn $p(S^\pi)$, which can be modeled autoregressively due to the sequential nature of $S^\pi$.
At inference time, we can sample $G$ without explicitly computing $p(G)$ by sampling $S^\pi$, which
maps to $G$ via $f_G$.

Given the above definitions, we can write $p(G)$ as the marginal distribution of the joint distribution $p(G,S^\pi)$:
$$p(G) = \sum_{S^\pi}{ p(S^\pi) \ \mathbf{1}[f_G(S^\pi)=G]},$$
where $p(S^\pi)$ is the distribution that we want to learn using a generative model.
Due to the sequential nature of $S^\pi$, we further decompose $p(S^\pi)$ as the product of conditional distributions over the elements: 
$$p(S^\pi) = \prod_{i=1}^{n+1}{p(S^\pi_i|S^\pi_1,...,S^\pi_{i-1})}.$$

$S^\pi_{n+1}$ is set to $\texttt{EOS}$ as the end of sequence to represent sequences of variable length.

#### The GraphRNN framework
So far we have transformed the modeling of $p(G)$ to modeling $p(S^{\pi})$, which we further decomposed into the product of conditional probabilities $p(S^\pi_i|S^\pi_{<i})$. Note that $p(S^\pi_i|S^\pi_{<i})$ is highly complex as it has to capture how node $\pi(v_i)$ links to previous nodes based on how previous nodes are interconnected among each other.
%
Here we propose to parameterize $p(S^\pi_i|S^\pi_{<i})$ using expressive neural networks to model the complex distribution. To achieve scalable modeling, we let the neural networks share weights across all time steps $i$.

In particular, we use an RNN that consists of a *state-transition function* and an *output function*:
$$
\begin{align}
  h_i &= f_{\mathrm{trans}}({h}_{i-1}, S^\pi_{i-1}),\\
  \theta_{i} &= f_{\mathrm{out}}({h}_i),
\end{align}
$$
where $ h_i \in \R^{d}$ is a vector that encodes the state of the graph generated so far, $S^\pi_{i-1}$ is the adjacency vector for the most recently generated node $i-1$, and $\theta_i$ specifies the distribution of next node's adjacency vector (i.e., $S^{\pi}_i \sim \mathcal{P}_{\theta_i}$).
In general, $f_{\mathrm{trans}}$ and $f_{\mathrm{out}}$ can be arbitrary neural networks, and $\mathcal{P}_{\theta_i}$ can be an arbitrary distribution over binary vectors. 
This general framework is summarized in the algorithm:
\begin{algorithm}[t]
\begin{algorithmic}
   \STATE {\bfseries Input:} RNN-based transition module $f_{trans}$, output module $f_{out}$, probability distribution $\mathcal{P}_{\theta_i}$ parameterized by $\theta_i$, start token $\texttt{SOS}$, end token $\texttt{EOS}$, empty graph state $h'$
   \STATE {\bfseries Output:} Graph sequence $S^{\pi}$
   \STATE $S^{\pi}_1=\texttt{SOS}$, $h_1 =  h'$, $i=1$ 
   \REPEAT
   \STATE $i=i+1$
   \STATE $h_i = f_{\mathrm{trans}}( h_{i-1},S^{\pi}_{i-1})$ \COMMENT{update graph state}
   \STATE $\theta_i = f_{\mathrm{out}}(h_i)$
   \STATE $S^{\pi}_i \sim \mathcal{P}_{\theta_i}$ \COMMENT{sample node $i$'s edge connections}
   \UNTIL{$S^{\pi}_i$ is $\texttt{EOS}$}
   \STATE {\bfseries Return} $S^{\pi}=(S^{\pi}_1,...,S^{\pi}_i)$
\end{algorithmic}
\end{algorithm}

#### GraphRNN variants
Different variants of the GraphRNN model correspond to different assumptions about $p(S^\pi_i|S^\pi_{<i})$. Recall that each dimension of $S^\pi_i$ is a binary value that models existence of an edge between the new node $\pi(v_{i})$ and a previous node $\pi(v_{j}),j\in \{1,...,i-1\}$.
We propose two variants of GraphRNN, both of which implement the transition function $f_{\mathrm{trans}}$ (i.e., the graph-level RNN) as a Gated Recurrent Unit (GRU) but differ in the implementation of $f_{\mathrm{out}}$ (i.e., the edge-level model).
Both variants are trained using stochastic gradient descent with a maximum likelihood loss over $S^\pi$ &mdash; i.e., we optimize the parameters of the neural networks to optimize $\prod{p_{model}(S^\pi)}$ over all observed graph sequences.
%further discuss some improvement of $S^\pi_i$'s design so that $p_{model}(S^\pi)$ can be more tractably learned.

##### Multivariate Bernoulli
First we present a simple baseline variant GraphRNN-S (``S'' for ``simplified''). 
In this variant, $p(S^\pi_i|S^\pi_{<i})$ is modeled as a multivariate Bernoulli distribution, parameterized by the $\theta_i\in \mathbb{R}^{i-1}$ vector that is output by $f_{\mathrm{out}}$. 
In particular, $f_{\mathrm{out}}$ is implemented as single layer MLP with
sigmoid activation function, that shares weights across all time steps. 
The output of $f_{\mathrm{out}}$ is a vector $\theta_i$, whose element $\theta_{i}[j]$ can be interpreted as a probability of edge $(i,j)$. 
Edges are then sampled in $S^\pi_i$  (independently) according to a multivariate Bernoulli distribution parametrized by $\theta_i$.

##### Dependent Bernoulli sequence
To fully capture complex edge dependencies, in the full model $p(S^\pi_i|S^\pi_{<i})$ is further decomposed into a product of conditionals,
$$
  p(S^\pi_i|S^\pi_{<i}) = \prod_{j=1}^{i-1} p(S^\pi_{i,j} | S^\pi_{i,<j}, S^\pi_{<i}),
$$
where $S^\pi_{i, j}$ denotes a binary scalar that is $1$ if node $\pi(v_{i+1})$ is connected to node $\pi(v_{j})$ (under ordering $\pi$). 
In this variant, each distribution in the product is approximated by an another RNN. 
Conceptually, we have a hierarchical RNN, where the first (i.e., the graph-level) RNN generates the nodes and maintains the state of the graph, while the second (i.e., the edge-level) RNN generates the edges of a given node. 
In our implementation, the edge-level RNN is a GRU model, where the hidden state is initialized via the graph-level hidden state $h_i$ and where the output at each step is mapped by a MLP to a scalar indicating the probability of having an edge. $S^\pi_{i, j}$ is sampled from this distribution specified by the $j$th output of the $i$th edge-level RNN, and is fed into the $j+1$th input of the same RNN. All edge-level RNNs share the same parameters.


#### Tractability via breadth-first search
**NB:** RNNs require fixed-size input vectors, while we previously defined $S^\pi_i$ as having varying dimensions depending on $i$. An efficient and flexible scheme to address this issue is based on breadth-first-search (BFS).

Rather than learning to generate graphs under any possible node permutation, BFS node orderings can be used to learn to generate graphs without a loss of generality. Formally, we modify the equation $$S^\pi=f_S(G,\pi)=(S^\pi_1, \ldots, S^\pi_{n})$$ to
$$S^\pi=f_S(G,\text{BFS}(G,\pi)),$$
where $\text{BFS}(\cdot)$ denotes the deterministic BFS function.
In particular, this BFS function takes a random permutation $\pi$ as input, picks $\pi(v_1)$ as the starting node and appends the neighbors of a node into the BFS queue in the order defined by $\pi$.
Note that the BFS function is many-to-one, i.e., multiple permutations can map to the same ordering after applying the BFS function. 

**Benefits of using BFS to specify the node ordering during generation:**
- We only need to train on all possible BFS orderings rather than on all possible node permutations. In other words, multiple node permutations map to the same BFS ordering, reducing the overall number of sequences we need to consider. In the worst case &mdash; for example, for star graphs &mdash; the number of BFS orderings is $n!$; however, we observe substantial reductions for many real-world graphs.  
- The BFS ordering makes learning easier by reducing the number of edge predictions we need to make in the edge-level RNN; in particular,  when we are adding a new node under a BFS ordering, the only possible edges for this new node are those connecting to nodes that are in the ``frontier'' of the BFS (i.e., nodes that are still in the BFS queue)---a notion formalized by Proposition: Suppose $v_1, \ldots, v_n$ is a BFS ordering of $n$ nodes in graph $G$, and $(v_i, v_{j-1}) \in E$ but $(v_i, v_j) \not \in E$ for some $i < j \le n$,  then $(v_{i'}, v_{j'}) \not \in E$, $\forall 1 \le i' \le i$ and $j \le j' < n$.

Importantly, this insight allows us to redefine the variable size $S^\pi_i$ vector as a fixed $M$-dimensional vector, representing the connectivity between node $\pi(v_i)$ and nodes in the current BFS queue with maximum size $M$:
$$S^\pi_i = (A^\pi_{\max(1,i-M),i}, \ldots,A^\pi_{i-1,i})^T, i\in \{2, \ldots, n\}.$$

We can bound $M$ as follows:
With a BFS ordering the maximum number of entries that GraphRNN needs to predict for $S^\pi_i$, $\forall 1 \le i \le n$ is 
$O\left(\max_{d=1}^{\mathrm{diam}(G)} \left|\left\{v_i | \mathrm{dist}(v_i, v_1) = d\right\} \right| \right)$,
where $\mathrm{dist}$ denotes the shortest-path-distance between vertices. 

The overall time complexity of GraphRNN is thus $O(Mn)$.


#### Model Capacity
This section analyzes the representational capacity of GraphRNN, illustrating how it is able to capture complex edge dependencies. 
In particular, we discuss two very different cases on how GraphRNN can learn to generate graphs with a *global community structure* as well as graphs with a very *regular geometric structure*.
For simplicity, we assume that $h_i$ (the hidden state of the graph-level RNN) can exactly encode $S^\pi_{<i}$, and that the edge-level RNN can encode $S^\pi_{i,<j}$.
That is, we assume that our RNNs can maintain memory of the decisions they make and elucidate the models capacity in this ideal case.
We similarly rely on the universal approximation theorem of neural networks.


##### Graphs with community structure
GraphRNN can model structures that are specified by a given probabilistic model. This is because the posterior of a new edge probability can be expressed as a function of the outcomes of previous nodes.
For instance, suppose that the training set contains graphs generated from the following distribution $p_{com}(G)$: half of the nodes are in community $A$, and half of the nodes are in community $B$ (in expectation), and nodes are connected with probability $p_s$ within each community and probability $p_d$ between communities.
Given such a model, we have the following key (inductive) observation:
\begin{observation}\label{obs:com}
Assume there exists a parameter setting for GraphRNN such that it can generate $S^\pi_{<i}$ and $S^\pi_{i,<j}$ according to the distribution over $S^\pi$  implied by  $p_{com}(G)$, then there also exists a parameter setting for GraphRNN such that it can output $p(S^\pi_{i, j} | S^\pi_{i, <j}, S^\pi_{<i})$ according to $p_{com}(G)$. 
\end{observation}
This observation follows from three facts:
First, we know that $p(S^\pi_{i, j} | S^\pi_{i, <j}, S^\pi_{<i})$ can be expressed as a function of $p_s$, $p_d$, and $p(\pi(v_j) \in A), p(\pi(v_j) \in B) \: \forall 1 \leq j \le i$ (which holds by $p_{com}$'s definition).
Second, by our earlier assumptions on the RNN memory, $S^\pi_{<i}$ can be encoded into the initial state of the edge-level RNN, and the edge-level RNN can also encode the outcomes of $S^\pi_{i,<j}$.
Third, we know that $p(\pi(v_i) \in A)$ is computable from $S^\pi_{<i}$ and $S^\pi_{i,1}$ (by Bayes' rule and $p_{com}$'s definition, with an analogous result for $p(\pi(v_i) \in B)$). 
Finally, GraphRNN can handle the  base case of the induction in Observation \ref{obs:com}, i.e., $S_{i,1}$, simply by sampling according to $0.5p_s + 0.5p_d$ at the first step of the edge-level RNN (i.e., 0.5 probability $i$ is in same community as node $\pi(v_1)$). 

##### Graphs with regular structure
GraphRNN can also naturally learn to generate regular structures, due to its ability to learn functions that only activate for $S^\pi_{i, j}$ where $v_j$ has specific degree. 
For example, suppose that the training set consists of ladder graphs.
To generate a ladder graph, the edge-level RNN must handle three key cases: if $\sum_{k=1}^{j}S^\pi_{i,j} = 0$, then the new node should only connect to the degree $1$ node or else any degree $2$ node; if $\sum_{k=1}^{j}S^\pi_{i,j} = 1$, then the new node should only connect to the degree $2$ node that is exactly two hops away; and finally, if $\sum_{k=1}^{j}S^\pi_{i,j} = 2$ then the new node should make no further connections. 
And note that all of the statistics needed above are computable from $S^\pi_{<i}$ and $S^\pi_{i,<j}$. 

---

### GraphVAE
### Generative Diffusion Models on Graphs
<!--
Some surveys:
- [Survey 1](https://arxiv.org/abs/2302.02591)
- [Survey 2](https://arxiv.org/abs/2401.15617v2)

Three paradigms of diffusion models:
- Score Matching with Langevin Dynamics (SMLD).
- Denoising Diffusion Probabilistic Model (DDPM).
- Score-based Generative Model (SGM).

#### EDP-GNN
[Original paper](https://arxiv.org/abs/2003.00638) <br>
It is the very first score matching based diffusion method for undirected graph generation.

This model can generate only adjacency matrices, not attributes.
-->

#### DiGress
[Original paper](https://arxiv.org/abs/2209.14734)<br>
This method extends the DDPM algorithm to
generate graphs with categorical node and edge attributes.

#### GraphARM
[Original paper](https://arxiv.org/abs/2307.08849)

<!--
#### GDSS (Graph Diffusion via the System of Stochastic differential equations)
[Original paper](https://proceedings.mlr.press/v162/jo22a/jo22a.pdf)
#### GSDM (Graph Spectral Diffusion Model)

### Applications of Generative Diffusion Models on Graphs
- Molecule generation (e. g., ...)
- Protein design (e.g., [Graph denoising diffusion for inverse protein folding](https://proceedings.neurips.cc/paper_files/paper/2023/file/20888d00c5df685de2c09790040e0327-Paper-Conference.pdf))
- Computer vision (e. g., ...)
- Multi-agent coordination ([Graph Diffusion for Robust Multi-Agent Coordination](https://raw.githubusercontent.com/mlresearch/v267/main/assets/zeng25f/zeng25f.pdf))
-->