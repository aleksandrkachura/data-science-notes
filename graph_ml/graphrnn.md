# GraphRNN: Autoregressive Generation of Graphs

**Original paper:** [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models](https://arxiv.org/abs/1802.08773)  
**Authors:** Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, and Jure Leskovec  
**Venue:** ICML 2018

## 1. Learning to generate graphs

A graph generative model learns a probability distribution over graphs and then samples new graphs from the learned distribution.

Given a dataset of graphs

$$
\mathcal{D}=\{G^{(1)},\ldots,G^{(N)}\},
$$

we assume that the graphs are sampled from an unknown data distribution $p_{\mathrm{data}}(G)$. The goal is to learn a parameterized model

$$
p_{\mathrm{model}}(G;\theta)\approx p_{\mathrm{data}}(G).
$$

There are two closely related objectives:

1. **Density estimation:** assign high probability to graphs resembling the training data.
2. **Sampling:** generate new graphs from $p_{\mathrm{model}}(G;\theta)$.

Graph generation has an additional difficulty that is absent, or less pronounced, in image and text generation: a graph does not have a canonical representation.

For example, permuting the rows and columns of an adjacency matrix does not change the underlying graph. If $G$ has $n$ nodes, then a node permutation $\pi$ produces an adjacency matrix $A^\pi$, and many different matrices can represent the same graph:

$$
A^{\pi_1}\neq A^{\pi_2},
\qquad
A^{\pi_1}\sim A^{\pi_2}.
$$

Here, $\sim$ means that the two adjacency matrices correspond to the same unlabeled graph.

Thus, a graph is not naturally represented by a single fixed vector. GraphRNN addresses this issue by converting a graph into a sequence whose form depends on a node ordering.

## 2. Autoregressive modeling

An autoregressive model represents a joint distribution using the chain rule. For a sequence $x_1,\ldots,x_T$,

$$p(x_1,\ldots,x_T) = \prod_{t=1}^{T} p(x_t\mid x_1,\ldots,x_{t-1}).$$

The model generates one element at a time. At step $t$, it predicts the next element conditioned on everything generated previously.

For example,

$$p(x_1,\ldots,x_T) = p(x_1)p(x_2\mid x_1)p(x_3\mid x_1,x_2)\cdots.$$

During training, the conditional distributions are fitted by maximum likelihood. Given training sequences $x^{(1)},\ldots,x^{(N)}$, we maximize

$$\mathcal{L}(\theta)=\sum_{k=1}^{N}\log p_{\mathrm{model}}(x^{(k)};\theta).$$

During generation, the model samples $x_1$, then $x_2$ conditioned on $x_1$, and so on.

GraphRNN applies this principle to a graph-construction sequence.

## 3. Turning a graph into a sequence

Consider an undirected graph $G=(V,E)$ with $n$ nodes. Choose an ordering of the nodes,

$$\pi(v_1),\pi(v_2),\ldots,\pi(v_n).$$

The ordering determines the order in which nodes will be generated.

At generation step $i$, node $\pi(v_i)$ is added to the partially constructed graph. Since all nodes $\pi(v_1),\ldots,\pi(v_{i-1})$ have already been generated, we only need to specify which of them are connected to the new node.

Define the adjacency vector

$$S_i^\pi = \left(S^\pi_{i,1}, S^\pi_{i,2},\ldots, S^\pi_{i,i-1}\right).$$

The $j$-th component is

$$S^\pi_{i,j}=
\begin{cases}
1, & \text{if }(\pi(v_i),\pi(v_j))\in E,\\
0, & \text{otherwise}.
\end{cases}$$

Therefore,

$$S^\pi_i\in\{0,1\}^{i-1}.$$

The first node has no previous nodes, so $S^\pi_1=\varnothing$. The complete graph is represented by

$$S^\pi=(S^\pi_1,S^\pi_2,\ldots,S^\pi_n).$$

For an undirected graph, the sequence uniquely determines the graph once the ordering is fixed.

### Example

Suppose

$$A^\pi=
\begin{pmatrix}
0&1&0&0\\
1&0&1&1\\
0&1&0&0\\
0&1&0&0
\end{pmatrix}.$$

Then

$$S^\pi_1=\varnothing,
\qquad S^\pi_2=(1),
\qquad S^\pi_3=(0,1),
\qquad S^\pi_4=(0,1,0).$$

The construction is:

1. Add node $1$.
2. Add node $2$, connected to node $1$.
3. Add node $3$, connected to node $2$.
4. Add node $4$, connected to node $2$.

The graph-generation problem has become a sequence-generation problem.

## 4. The two-level sequence

The sequence $S^\pi$ has a hierarchical structure:

$$S^\pi=(S^\pi_1,S^\pi_2,\ldots,S^\pi_n),$$

where every $S^\pi_i$ is itself a sequence or vector of edge decisions.

### Node level

The node-level process decides whether another node should be generated. Each node-level step adds one new node.

### Edge level

After a new node is added, the edge-level process determines its connections to previously generated nodes.

Thus, graph generation consists of:

1. Generate a new node.
2. Generate the edges incident to that node.
3. Repeat until the graph is complete.

This motivates two components:

- a **node-level RNN**, which summarizes the graph generated so far;
- an **edge-level RNN**, which generates the edges of the current node.

The node-level RNN provides the initial state of the edge-level RNN. After the edge-level sequence has been generated, its final state is used to update the node-level RNN before the next node is generated.

## 5. Autoregressive factorization

The probability of a graph-generation sequence is factorized as

$$p(S^\pi)=\prod_{i=1}^{n}p\left(S^\pi_i\mid S^\pi_{<i}\right),$$

where

$$S^\pi_{<i}=(S^\pi_1,\ldots,S^\pi_{i-1}).$$

Because graphs may have different numbers of nodes, an end-of-sequence symbol is introduced:

$$S^\pi_{n+1}=\texttt{EOS}.$$

Therefore,

$$p(S^\pi)=\prod_{i=1}^{n+1}p\left(S^\pi_i\mid S^\pi_{<i}\right).$$

The EOS symbol indicates that no additional node should be generated.

It is important to distinguish between a node with no edges and termination of the entire graph. The sequence encoding must represent these events separately.

## 6. Graph likelihood and node orderings

A graph can be represented by many node orderings. If $\Pi$ denotes the set of permutations of $n$ nodes, then

$$|\Pi|=n!.$$

The same graph can correspond to many sequences:

$$S^{\pi_1},S^{\pi_2},\ldots,S^{\pi_{n!}}.$$

If the ordering is sampled uniformly,

$$\pi\sim\mathrm{Uniform}(\Pi),$$

and the model learns the distribution of the corresponding sequence $S^\pi$.

At generation time:

1. sample a sequence $S^\pi$;
2. convert it into an adjacency matrix;
3. interpret the adjacency matrix as a graph.

The graph distribution induced by the sequence model is

$$p_{\mathrm{model}}(G) = \sum_{S^\pi}p_{\mathrm{model}}(S^\pi) \mathbb{I}\!\left[f_G(S^\pi)=G\right],$$

where $f_G$ reconstructs a graph from its sequence.

In practice, GraphRNN does not enumerate all permutations. It uses sampled orderings and BFS-based orderings to make the problem tractable.

## 7. GraphRNN architecture

Let $h_i$ denote the hidden state of the node-level RNN before generating node $i$:

$$h_i = f_{\mathrm{node}}(h_{i-1},S^\pi_{i-1}).$$

The hidden state $h_i$ summarizes the partial graph generated before node $i$.

The edge-level RNN is initialized from $h_i$:

$$z_{i,0}=g_{\mathrm{init}}(h_i),$$

where $z_{i,j}$ is its hidden state while deciding the $j$-th edge of node $i$.

The edge-level recurrence is

$$z_{i,j}=f_{\mathrm{edge}}(z_{i,j-1},x_{i,j-1}),$$

where $x_{i,j-1}$ is a start token or the previous binary edge decision.

The output layer gives

$$\hat p_{i,j}=\sigma(W_{\mathrm{edge}}z_{i,j}+b_{\mathrm{edge}}),$$

and

$$S^\pi_{i,j}\sim\mathrm{Bernoulli}(\hat p_{i,j}).$$

The two recurrent modules share parameters across positions. This allows them to process graphs with different numbers of nodes and edges.

## 8. GraphRNN-S: independent edge decisions

The simplified model, GraphRNN-S, models the adjacency vector of a new node with a multivariate Bernoulli distribution:

$$p(S^\pi_i\mid S^\pi_{<i}) = \prod_{j=1}^{i-1}p(S^\pi_{i,j}\mid S^\pi_{<i}).$$

The node-level RNN produces $h_i$, and an output network predicts the Bernoulli parameters:

$$\theta_i=\sigma(Wh_i+b).$$

The component $\theta_{i,j}$ represents

$$\theta_{i,j}\approx p(S^\pi_{i,j}=1\mid S^\pi_{<i}).$$

The edge decisions are conditionally independent given the node-level hidden state:

$$p(S^\pi_i\mid S^\pi_{<i}) = \prod_{j=1}^{i-1} \theta_{i,j}^{S^\pi_{i,j}} (1-\theta_{i,j})^{1-S^\pi_{i,j}}.$$

This model is simple but cannot explicitly condition edge $j$ on decisions for edges $1,\ldots,j-1$ of the same node.

## 9. GraphRNN: dependent edge generation

The full GraphRNN model introduces a second RNN for the edge sequence of each new node:

$$p(S^\pi_i\mid S^\pi_{<i}) = \prod_{j=1}^{i-1} p\left(S^\pi_{i,j}\mid S^\pi_{i,<j},S^\pi_{<i}\right),$$

where

$$S^\pi_{i,<j}=(S^\pi_{i,1},\ldots,S^\pi_{i,j-1}).$$

The edge-level RNN is initialized from the node-level state:

$$z_{i,0}=g_{\mathrm{init}}(h_i).$$

Then, for $j=1,\ldots,i-1$,

$$z_{i,j}=f_{\mathrm{edge}}(z_{i,j-1},x_{i,j-1}),$$

$$\hat p_{i,j}=\sigma(Wz_{i,j}+b),$$

$$S^\pi_{i,j}\sim\mathrm{Bernoulli}(\hat p_{i,j}).$$

The model can therefore use both:

1. the previously generated graph, summarized by $h_i$;
2. previous edge decisions for the current node, summarized by $z_{i,j}$.

## 10. Training with teacher forcing

During training, the target graph sequence is known. The model uses **teacher forcing**: the ground-truth previous symbol is supplied as the next input.

If

$$S^\pi_i=(0,1,1,0),$$

then the edge-level RNN receives

$$\texttt{SOS},0,1,1$$

and predicts

$$0,1,1,0.$$

For a Bernoulli prediction with target $y\in\{0,1\}$ and predicted probability $\hat y$, the loss is

$$\ell_{\mathrm{BCE}} = -\left[y\log\hat y+(1-y)\log(1-\hat y)\right].$$

The total negative log-likelihood is

$$\mathcal{L}(\theta) = -\sum_{i=1}^{n+1} \log p_\theta(S^\pi_i\mid S^\pi_{<i}).$$

For the full model,

$$\mathcal{L}(\theta) = -\sum_{i=1}^{n+1}\sum_j \log p_\theta(S^\pi_{i,j} \mid S^\pi_{i,<j},S^\pi_{<i}),$$

with masking for nonexistent positions.

During training:

$$x_{t-1}=\text{ground-truth previous output}.$$

During generation:

$$x_{t-1}=\text{sampled previous output}.$$

This difference creates exposure bias: an early generation error can affect all subsequent predictions.

## 11. Sampling a graph

A simplified generation procedure is:

1. Initialize the node-level RNN.
2. Generate the first node.
3. For each subsequent node:
   - update the node-level hidden state;
   - initialize the edge-level RNN;
   - sequentially predict edges to previous nodes;
   - sample binary edge decisions;
   - update the node-level state.
4. Stop when EOS is generated.
5. Return the constructed graph.

```text
h_node <- initial state
S <- empty graph

repeat:
    h_node <- NodeRNN(h_node, representation of previous node)
    h_edge <- initialize EdgeRNN from h_node

    for each eligible previous node j:
        p_j <- EdgeOutput(h_edge)
        x_j ~ Bernoulli(p_j)
        add edge if x_j = 1
        h_edge <- EdgeRNN(h_edge, x_j)

    update S with the new node and its edges

until EOS is generated

return S
```

The implementation must distinguish between a node with no incident edges and termination of the graph.

## 12. Why the naive formulation is expensive

In the naive formulation, node $i$ may make $i-1$ edge decisions. The total number of decisions is

$$\sum_{i=1}^{n}(i-1)=\frac{n(n-1)}{2}=O(n^2).$$

Long edge sequences also make learning difficult because:

- the edge-level RNN must preserve information over many steps;
- many edge decisions may be impossible given the ordering;
- the same graph has many possible orderings;
- autoregressive errors accumulate.

GraphRNN uses BFS-based orderings to reduce the effective sequence length.

## 13. BFS-based node ordering

A breadth-first search ordering is obtained by:

1. choosing a starting node;
2. placing it in a queue;
3. repeatedly removing the next node;
4. appending its unvisited neighbors;
5. resolving ties according to a chosen order.

BFS tends to place nearby nodes close together. Once a node and its neighborhood have been processed, later nodes often cannot connect back to it without violating the BFS ordering.

Consequently, the edge-level model need not inspect all previous nodes.

### Truncated adjacency vectors

GraphRNN uses a fixed maximum look-back size $M$. Instead of

$$S^\pi_i=(A^\pi_{i,1},\ldots,A^\pi_{i,i-1}),$$

use

$$S^\pi_i = \left(A^\pi_{i,\max(1,i-M)},\ldots,A^\pi_{i,i-1}\right).$$

After padding, every edge-level input has dimension $M$.

The value of $M$ is related to the maximum size of a BFS frontier. In the worst case, $M$ can still be large, but for many sparse graphs it is much smaller than $n$.

The approximate generation complexity becomes

$$O(Mn)$$

instead of $O(n^2)$.

BFS does not eliminate permutation sensitivity, but it reduces the number of relevant orderings and often reduces the effective look-back distance.

## 14. What the model can represent

Assume that:

- the node-level RNN can encode $S^\pi_{<i}$;
- the edge-level RNN can encode $S^\pi_{i,<j}$.

Then the model can approximate

$$p(S^\pi_{i,j}\mid S^\pi_{i,<j},S^\pi_{<i}).$$

This allows both local and non-local dependencies.

### Community structure

Suppose a graph is generated by a two-community stochastic block model:

- each node belongs to community $A$ or $B$;
- within-community edges appear with probability $p_s$;
- between-community edges appear with probability $p_d$.

The probability of a new edge depends on the likely community memberships of its endpoints. Evidence for these memberships can be inferred from the previously generated graph.

The node-level state can summarize evidence about previous nodes, while the edge-level state tracks connections already made by the current node. The model can therefore approximate the conditional edge probabilities induced by latent community assignments.

### Regular structures

Consider ladder graphs or cycles. The decision to connect the new node to a candidate previous node may depend on:

- the candidate's degree;
- whether the new node already has one or two edges;
- the distance to recently generated nodes;
- whether the boundary of the partial graph has been reached.

These quantities can be represented implicitly by the recurrent states. Thus, GraphRNN can learn structured graph-construction rules rather than only independent edge probabilities.

## 15. Strengths and limitations

### Strengths

- Direct likelihood-based training.
- Supports graphs with a variable number of nodes.
- Captures dependencies at node and edge levels.
- BFS ordering improves scalability.
- Produces a distribution over graphs rather than a single deterministic graph.
- Does not require a manually designed graph-generation mechanism.

### Limitations

- Sensitive to node ordering.
- The same graph can correspond to many training sequences.
- Autoregressive sampling is sequential and difficult to parallelize.
- Errors accumulate during generation.
- Long-range dependencies may be difficult for RNNs.
- BFS truncation is approximate when $M$ is too small.
- Node and edge attributes require extensions.
- Structural validity constraints are not automatically enforced.
- Maximum likelihood does not directly optimize graph-level properties.

## 16. Evaluation

Likelihood alone is not sufficient for graph generation. A model can assign high likelihood to sequences while failing to reproduce important graph-level properties.

A generated graph set can be compared with a real graph set using:

- degree distributions;
- clustering coefficients;
- motif counts;
- numbers of connected components;
- shortest-path statistics;
- orbit counts;
- assortativity;
- community statistics.

For a graph statistic $\phi(G)$, compare

$$\{\phi(G_1),\ldots,\phi(G_N)\}$$

for real graphs with

$$\{\phi(\widetilde G_1),\ldots,\phi(\widetilde G_M)\}$$

for generated graphs. Distances such as maximum mean discrepancy can then be used to compare the two distributions.

A good generator should reproduce the distribution of structural properties, not merely produce plausible individual graphs.