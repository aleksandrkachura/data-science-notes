# GraphVAE: Generation of Small Graphs Using Variational Autoencoders

**Original paper:** [GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders](https://arxiv.org/abs/1802.03480)  
**Authors:** Martin Simonovsky and Nikos Komodakis  
**Venue:** International Conference on Artificial Neural Networks (ICANN), 2018

## 1. Motivation

Graph generative models learn a distribution over graphs and generate new graph instances from the learned distribution.

This problem appears in settings involving:

- molecular and chemical graphs;
- social and interaction networks;
- communication and infrastructure networks;
- software dependency graphs;
- synthetic data generation;
- graph benchmarking and simulation.

Graphs are difficult to generate because they are discrete, combinatorial objects with no canonical node ordering. A graph can be represented by many different adjacency matrices, all describing the same underlying structure.

GraphVAE studies a different strategy from autoregressive models such as GraphRNN. Instead of generating nodes and edges sequentially, GraphVAE encodes a graph into a continuous latent vector and decodes the complete graph in one step.

The central idea is

$$
G\longrightarrow z\longrightarrow \widehat G,
$$

where $z$ is a continuous latent representation.

## 2. Variational autoencoder refresher

Let $G$ denote a graph and $z$ a latent variable. The generative model is

$$
p_\theta(G,z)=p(z)p_\theta(G\mid z),
$$

where the prior is commonly chosen as

$$
 p(z)=\mathcal{N}(0,I).
$$

The posterior

$$
 p_\theta(z\mid G)
$$

is generally intractable. GraphVAE introduces an approximate posterior

$$
 q_\phi(z\mid G).
$$

Using the evidence lower bound,

$$
\log p_\theta(G)
\geq
\mathbb{E}_{q_\phi(z\mid G)}
\left[\log p_\theta(G\mid z)\right]
-
D_{\mathrm{KL}}
\left(q_\phi(z\mid G)\,\|\,p(z)\right).
$$

Therefore, the negative VAE objective is

$$
\mathcal{L}_{\mathrm{VAE}}
=
-\mathbb{E}_{q_\phi(z\mid G)}
\left[\log p_\theta(G\mid z)\right]
+
D_{\mathrm{KL}}
\left(q_\phi(z\mid G)\,\|\,p(z)\right).
$$

The two terms have different roles:

- the reconstruction term makes $z$ informative about $G$;
- the KL term regularizes the latent distribution toward the prior;
- the regularized latent space can be sampled at generation time.

For a diagonal Gaussian posterior,

$$
q_\phi(z\mid G)
=
\mathcal{N}
\left(
\mu_\phi(G),
\operatorname{diag}(\sigma_\phi^2(G))
\right),
$$

we use the reparameterization trick:

$$
 z=\mu_\phi(G)+\sigma_\phi(G)\odot\epsilon,
 \qquad
 \epsilon\sim\mathcal{N}(0,I).
$$

## 3. Graph representation

GraphVAE assumes a maximum number of nodes $N_{\max}$. A graph is represented by node features and adjacency information.

The node-feature matrix is

$$
X\in\mathbb{R}^{N_{\max}\times d_x},
$$

where each row represents a node. In a molecular graph, node features may encode atom types.

For a simple untyped graph, the adjacency matrix is

$$
A\in\{0,1\}^{N_{\max}\times N_{\max}}.
$$

For a graph with multiple edge types, the adjacency representation is a tensor:

$$
A\in\{0,1\}^{N_{\max}\times N_{\max}\times d_e}.
$$

Here $A_{i,j,k}=1$ indicates that nodes $i$ and $j$ are connected by edge type $k$.

If a graph has fewer than $N_{\max}$ nodes, it is padded. The model must distinguish real nodes from padding nodes, usually through an additional node category or a mask.

The graph representation can therefore be written as

$$
G=(X,A).
$$

The decoder predicts this representation directly rather than generating it node by node.

## 4. The permutation problem

The matrix representation depends on node ordering. If $P$ is a permutation matrix, then

$$
X' = PX,
\qquad
A'=PAP^\top
$$

represent the same unlabeled graph.

A graph-generating model should therefore treat

$$
(X,A)\quad\text{and}\quad(PX,PAP^\top)
$$

as equivalent.

However, a standard elementwise reconstruction loss treats them as different targets. A decoder may produce a graph isomorphic to the target while receiving a large loss because its node ordering differs.

A permutation-aware reconstruction objective can be written as

$$
\mathcal{L}_{\mathrm{rec}}(G,\widehat G)
=
\min_{P\in\mathcal{P}_{N}}
\ell\left(G,P\widehat G P^\top\right),
$$

where $\mathcal{P}_N$ is the set of permutation matrices.

Exact minimization over all $N!$ permutations is infeasible. GraphVAE therefore uses an approximate graph-matching procedure to align the predicted graph with the target graph before computing the reconstruction loss.

This is a central difference from ordinary image VAEs: graph reconstruction requires handling equivalence under permutations.

## 5. GraphVAE encoder

The encoder maps a graph to the parameters of a latent Gaussian distribution:

$$
q_\phi(z\mid G)
=
\mathcal{N}
\left(
\mu_\phi(G),
\operatorname{diag}(\sigma_\phi^2(G))
\right).
$$

A graph neural network first produces node representations:

$$
H^{(0)}=X,
$$

$$
H^{(\ell+1)}
=
\operatorname{GNN}^{(\ell)}(H^{(\ell)},A).
$$

After several message-passing layers, a graph-level pooling operation produces a graph representation:

$$
 h_G=\operatorname{READOUT}(H^{(L)}).
$$

The parameters of the approximate posterior are predicted by two neural networks:

$$
\mu_\phi(G)=f_\mu(h_G),
$$

$$
\log\sigma_\phi^2(G)=f_\sigma(h_G).
$$

The encoder can be summarized as

$$
G\longrightarrow h_G\longrightarrow
\left(\mu_\phi(G),\sigma_\phi(G)\right).
$$

A desirable property is permutation invariance:

$$
\operatorname{Enc}(PGP^\top)=\operatorname{Enc}(G).
$$

Message passing followed by permutation-invariant pooling is a natural way to obtain this property.

## 6. GraphVAE decoder

The decoder maps a latent vector to the complete graph representation:

$$
 z\longrightarrow(\widehat X,\widehat A).
$$

Typical outputs are

$$
\widehat X\in\mathbb{R}^{N_{\max}\times d_x}
$$

and

$$
\widehat A\in\mathbb{R}^{N_{\max}\times N_{\max}\times d_e}.
$$

For a node category, the decoder predicts a categorical distribution:

$$
 p_\theta(X_i\mid z)
=\operatorname{Categorical}
\left(\operatorname{softmax}(u_i(z))\right).
$$

For an edge type,

$$
 p_\theta(A_{ij}\mid z)
=\operatorname{Categorical}
\left(\operatorname{softmax}(v_{ij}(z))\right).
$$

For a binary edge,

$$
 p_\theta(A_{ij}=1\mid z)=\sigma(v_{ij}(z)).
$$

The decoder predicts many entries in parallel. This is faster than autoregressive sampling but produces a dense output whose size grows as

$$
O(N_{\max}^2d_e).
$$

This makes the original model appropriate mainly for small graphs.

## 7. Reconstruction loss

A simplified reconstruction loss combines node and edge losses:

$$
\mathcal{L}_{\mathrm{rec}}
=\lambda_X\mathcal{L}_X+\lambda_A\mathcal{L}_A.
$$

For node features,

$$
\mathcal{L}_X
=-\sum_i\log p_\theta(X_i\mid z).
$$

For edge types,

$$
\mathcal{L}_A
=-\sum_{i,j}\log p_\theta(A_{ij}\mid z).
$$

The complete VAE objective is

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{rec}}
+
\beta D_{\mathrm{KL}}
\left(q_\phi(z\mid G)\,\|\,p(z)\right).
$$

For a diagonal Gaussian posterior and standard normal prior,

$$
D_{\mathrm{KL}}(q_\phi(z\mid G)\|p(z))
=
-\frac{1}{2}\sum_{k=1}^{d_z}
\left(1+\log\sigma_k^2-\mu_k^2-\sigma_k^2\right).
$$

In graph data, the number of absent edges can greatly exceed the number of present edges. A decoder can therefore obtain a deceptively good loss by predicting no edge almost everywhere.

Practical remedies include:

- weighting positive and negative edges differently;
- masking padded nodes;
- using class-balanced cross-entropy;
- evaluating graph validity separately from reconstruction accuracy;
- explicitly enforcing adjacency symmetry.

## 8. Graph matching and training

A training example consists of a graph represented by $(X,A)$. The encoder produces $q_\phi(z\mid G)$, and a latent vector is sampled by reparameterization.

The decoder produces $(\widehat X,\widehat A)$. Since the predicted node ordering may differ from the target ordering, a graph-matching step estimates an alignment between them.

The reconstruction loss is then computed after alignment:

$$
\mathcal{L}_{\mathrm{rec}}
=
\ell\left((X,A),P(\widehat X,\widehat A)P^\top\right).
$$

The overall training pipeline is

$$
G
\xrightarrow{\mathrm{encoder}}
q_\phi(z\mid G)
\xrightarrow{\mathrm{sampling}}
z
\xrightarrow{\mathrm{decoder}}
(\widehat X,\widehat A)
\xrightarrow{\mathrm{matching}}
\mathcal{L}_{\mathrm{rec}}.
$$

The matching operation is an approximation. Its quality affects both the stability of training and the quality of the learned latent space.

## 9. Sampling new graphs

After training, a new graph is generated as follows:

1. Sample a latent vector:
   $$
   z\sim\mathcal{N}(0,I).
   $$
2. Decode $z$ into node and edge distributions.
3. Sample node types and edge types.
4. Remove padding nodes.
5. Remove self-loops if required.
6. Symmetrize the adjacency matrix if necessary.
7. Check graph validity.
8. Return the generated graph.

The important distinction is that all node and edge positions can be predicted in parallel after sampling $z$.

## 10. Molecular graph generation

For molecular graphs, node features may represent atom types and edge features may represent bond types.

A molecular generation pipeline is:

1. sample $z$;
2. predict atom categories;
3. predict bond categories;
4. construct the graph;
5. remove inactive nodes;
6. sanitize the molecule with a cheminformatics toolkit;
7. evaluate chemical validity and desired properties.

A decoded graph is not automatically a valid molecule:

$$
\text{decoded graph}\neq\text{chemically valid molecule}.
$$

Possible problems include:

- impossible atom valence;
- invalid combinations of atom and bond types;
- disconnected fragments;
- inconsistent aromaticity;
- invalid hydrogen counts;
- self-loops or multiple incompatible edges.

Useful evaluation metrics include:

- validity;
- uniqueness;
- novelty;
- reconstruction accuracy;
- connectedness;
- distribution of molecular properties.

The molecular setting is one benchmark for GraphVAE, but the underlying method is a general model for small attributed graphs.

## 11. Latent-space interpretation

The VAE imposes a regularized continuous latent space. This allows several operations.

### Sampling

Sample

$$
 z\sim\mathcal{N}(0,I)
$$

and decode it to obtain a graph.

### Encoding

Given a graph $G$, use the encoder mean as a deterministic representation:

$$
 z_G=\mu_\phi(G).
$$

This representation can be used for visualization, clustering, or graph retrieval.

### Interpolation

Given graphs $G_0$ and $G_1$, encode them as $z_0$ and $z_1$ and interpolate:

$$
 z(t)=(1-t)z_0+tz_1,
 \qquad t\in[0,1].
$$

Decode intermediate vectors to obtain graph candidates.

Interpolation is not guaranteed to produce valid or semantically smooth graph sequences. It is an empirical property that must be evaluated.

### Property optimization

If a graph-level property predictor $r(z)$ is available, one can search for latent vectors with desirable values:

$$
\max_z r(z).
$$

This can be done with gradient-based optimization, Bayesian optimization, or evolutionary search, depending on the application.

## 12. Practical limitations

### Fixed maximum graph size

The decoder is designed for $N_{\max}$ nodes. Graphs larger than this limit cannot be represented without modifying the architecture.

### Dense output

The adjacency tensor has quadratic size:

$$
O(N_{\max}^2d_e).
$$

This becomes expensive for large graphs.

### Permutation ambiguity

The model must align predicted and target graphs or use a permutation-invariant objective. Approximate matching can be computationally expensive and imperfect.

### Invalid graphs

Independent node and edge predictions may violate structural constraints. The model does not automatically guarantee connectivity, valence, or other domain-specific validity properties.

### Class imbalance

Most potential edges may be absent. Without appropriate weighting, the reconstruction objective can be dominated by negative edge examples.

### Posterior collapse

If the decoder is sufficiently powerful, it may ignore $z$. Then

$$
q_\phi(z\mid G)\approx p(z)
$$

and the latent representation contains little information about the graph. KL annealing, a smaller decoder, or alternative VAE objectives may be needed in some implementations.

### Sampling quality

A low reconstruction loss does not guarantee that random samples from the prior are valid or realistic. Reconstruction, prior sampling, and graph-level distribution matching should be evaluated separately.

## 13. GraphVAE versus GraphRNN

GraphRNN uses an explicit generation order and models

$$
 p(S^\pi)=\prod_i p(S^\pi_i\mid S^\pi_{<i}).
$$

GraphVAE models a latent-variable distribution:

$$
 p_\theta(G)=\int p_\theta(G\mid z)p(z)\,dz.
$$

GraphRNN has a natural mechanism for variable graph sizes and sequential constraints, but its sampling process is slow and order-sensitive.

GraphVAE decodes in parallel and provides a continuous latent space, but it requires a fixed maximum size and must solve permutation matching.

Neither model is universally better. The appropriate choice depends on the application:

- use autoregressive generation when sequential structure and explicit conditional decisions are important;
- use a VAE when latent representations, interpolation, and parallel decoding are important.

## 16. Evaluation

GraphVAE should be evaluated at several levels.

### Reconstruction

Can the model reconstruct held-out graphs after accounting for node permutations?

### Validity

Are generated graphs structurally valid? In domain-specific settings, are they chemically, syntactically, or physically valid?

### Diversity

Do samples differ from each other, or does the model produce near-duplicates?

### Novelty

Are generated graphs different from the training examples?

### Distributional similarity

Do generated graphs reproduce statistics of the data distribution, such as:

- degree distributions;
- clustering coefficients;
- motif counts;
- connected-component statistics;
- shortest-path distributions;
- node and edge-type frequencies?

A generative model should not be evaluated only by elementwise adjacency accuracy. Graph-level and domain-specific metrics are essential.

<!--
## 17. Suggested practical exercises

### Exercise 1: permutation-aware reconstruction

Take a graph $G$ and apply a random permutation $P$:

$$
A' = PAP^\top.
$$

Verify that $A$ and $A'$ represent the same graph. Compare:

1. elementwise adjacency loss;
2. loss after alignment;
3. a graph-isomorphism-based comparison.

### Exercise 2: implement a small GraphVAE

For a dataset of graphs with at most $N_{\max}$ nodes:

1. pad node features and adjacency matrices;
2. implement a message-passing encoder;
3. predict $\mu$ and $\log\sigma^2$;
4. sample $z$ using the reparameterization trick;
5. decode node and edge logits;
6. compute masked reconstruction and KL losses;
7. sample graphs from the prior;
8. evaluate validity and graph statistics.

### Exercise 3: latent interpolation

Choose two graphs $G_0$ and $G_1$, encode them as $z_0$ and $z_1$, and decode points along

$$
 z(t)=(1-t)z_0+tz_1.
$$

Record whether intermediate graphs are valid and whether their structural properties change smoothly.
-->