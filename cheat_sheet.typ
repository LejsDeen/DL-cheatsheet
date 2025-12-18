#import "template.typ": *
#import "@preview/diagraph:0.2.1": *
#import "@preview/xarrow:0.4.0": xarrow

// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.
#show: project.with()

#set math.mat(delim: "[")

= Connectionism

== Perceptron

*Threshold Unit* 
$f[w, b](x) = "sign"(x dot  w + b)$ with *Dec. Boundary* $x dot w + b =^! 0 <=> (x  dot  w) / norm(w) + b / norm(w) =^! 0$.

*Geometric Margin* $gamma[w, b](x, y) = (y (x dot  w + b)) / norm(w)$.\
*Maximum Margin Classifier*\
$(w^"*", b^"*") in "argmax"_(w,b) gamma[w, b](cal(S))$, \
with $gamma[w, b](cal(S)) := min_((x,y) in cal(S)) gamma[w, b](x, y)$.

#colorbox(title: [Perception Learning], color: silver, inline:false)[
  If $f[w, b](x) != y$: update $w$ `+=` $y x$, and $b$ `+=` $y$. \
$w_0 in "span"(x_1, ..., x_s) => w_t in "span"(x_1, ..., x_s) forall t$.
]

*Convergence* \
1. #text(0.89em)[If $exists w^*$, $norm(w^*) = 1$, s.t. $gamma[w^*](cal(S)) = gamma > 0 => w_t  dot  w^* >= t gamma$.]
2. Let $R = max_(x in S) norm(x)$. Then $ norm(w_t) <= R sqrt(t)$. \
#v(-6pt)
#text(0.9em)[$cos angle(w^*, w_t) = (w^*  dot  w_t) / (norm(w^*) norm(w_t)) >= (t gamma) / (sqrt(t) R) = sqrt(t) gamma / R <=^! 1 => t <= R^2 / gamma^2$.]

#colorbox(title: [Cover's Theorem for $cal(S) subset RR^n, |cal(S)| = s$],color: silver, inline:false)[
  $C(cal(S), n)$: \# of ways to separate $cal(S)$ in $n$ dimensions. Position of pts does not matter (general positition).

  $C(s + 1, n) = 2 sum_(i=0)^(n-1) binom(s, i)$, 
  $C(s, n) = 2^s$ for $s <= n$.

  Phase transition at $s = 2n$. For $s < 2n$ empty version space is the exception, otherwise the rule.
]

=== Hopfield Networks

*Hopfield Model* 
$E(X) = -1/2 sum_(i != j) w_(i j) X_i X_j + sum_i b_i X_i$,
where $X_i in {plus.minus 1}$. $w_(i j) = w_(j i)$ , $w_(i i) = 0$.

#colorbox(title: [Hebbian Learning], color: silver, inline:false)[
  Choose patterns ${x}^s_(t=1) in {plus.minus 1}^n$, build weights once using them:
$w_(i j) = 1/n sum_(t=1)^s x_i^t x_j^t$, $w_(i i) = 0$. For inference, update $X$ iteratively: $X_i^(t+1) = "sign"(sum_j w_(i j) X_j^t + b_i)$ asynchronously.
Capacity for random, uncorrelated patterns: $s_"max" approx 0.138 n$.
]


== Feedforward Networks

=== Linear Models

*Linear regression* (MSE)\
$L[w](X, y) =  norm(X w - y)^2/(2n)$,
$nabla L = (X^top X w - X^top y) / n$.

*Moore-Penrose inverse solution* \
$w^* = X^* y in "argmin"_w L[w](X, y)$,
where $X^* = lim_(delta -> 0) (X^top X + delta I)^(-1) X^top$ Moore-Penrose inverse.

*Stochastic gradient descent update* \
$w_(t+1) = w_t + eta (y_(i_t) - w_t^top x_(i_t)) x_(i_t)$, $i_t ~ cal(U)([1, n])$.

*Gaussian noise model* \
$y_i = w^top x_i + epsilon_i$, $epsilon_i ~ "N"(0, sigma^2)$, LSQ equivalent to NLL of gaussian noise model. 

*Ridge regression* \
$h_lambda [w] = h[w] + lambda/2 norm(w)^2$,
$w^"*" = (X^top X + lambda I)^(-1) X^top y$.


*Logistic function* \
$sigma(z) = 1/(1 + e^(-z))$, $sigma(z) + sigma(-z) = 1$. \
$sigma' = sigma(1 - sigma)$, $sigma'' = sigma(1 - sigma)(1 - 2 sigma)$ \
*Cross entropy loss* for $y in {0, 1}$\
$ell(y, z) = -y log sigma(z) - (1 - y) log(1 - sigma(z))$ \
$= -log sigma((2y - 1)z)$.

*Logistic regression with CE loss*:
$L[w]=1/n sum_(i=1)^n ell_i (y_i , w^top x_i) $,$nabla ell_i = [sigma(w^top x_i) - y_i] x_i$.


#line(length: 100%)

=== Feedforward Networks

*Generic feedforward layer definition* \
$F: RR^(m(n+1)) times RR^n -> RR^m$, \
$F[theta](x) := phi(W x + b)$, $theta := "vec"(W, b)$

*Composition of layers* \
$G = F^L[theta^L] @ ... @ F^1[theta^1]$ \
where $F^l[W^l, b^l](x) := phi^l(W^l x + B^l)$

*Layer activations* \
$x^l := (F^l @ ... @ F^1)(x) = F^l(x^(l-1))$ \
identifying $x^0 = x$, $x^L = F(x)$

*Softmax function* \
$"softmax"(z)_i = e^(z_i) / sum_j e^(z_j)$, $"softmax"(A)_(i j) = e^(A_(i j)) / sum_k e^(A_(i k))$ \
$ell(y, z) = (-z_y + log sum_j e^(z_j)) / (ln 2)$

*Residual layer definition* \
$F[W, b](x) = x + (phi(W x + b) - phi(0))$ \
therefore $F[0, 0] = id$ \
Skip connection: Concatenate previous layer back in

=== Sigmoid Networks

*Sigmoid activation* \
$phi(z) := sigma(z) = 1/(1 + e^(-z))$

*Hyperbolic tangent activation* \
$tanh(z) = (e^z - e^(-z))/(e^z + e^(-z)) = 2 sigma(2z) - 1$ \
$tanh'(z) = 1 - tanh^2(z)$

*Baron's Theorem: Approximation error* \
For $f$ with finite $C_f := integral norm(omega) |hat(f)(omega)| dif omega < oo$ there exists MLP $g$ with one hidden layer of width $m$ that: \
$integral_B (f(x) - g_m(x))^2 mu(dif x) <= O(1/m)$

=== ReLU Networks

*ReLU activation* \
$phi(z) := (z)_+ := max{0, z}$

ReLU networks are universal function approximators

*Zalabsky: Connected regions* \
$R(H) <= sum_(i=0)^(min{n,m}) binom(m, i) := R(m)$

*Montufar: Connected regions in ReLU network* \
$R(m, L) >= R(m) (m/n)^(n(L-1))$, $L$: layers, $m$: width

== Gradient-Based Learning

=== Backpropagation

*Parameter derivatives for ridge function layers* \
$(partial x_i^l) / (partial w_(i j)^l) = dot(phi)_i^l x_j^(l-1)$, \
$dot(phi)_i^l := dot(phi)^l ((w_i^l)^top x^(l-1) + b_i^l)$ \
$(partial x_i^l) / (partial b_i^l) = dot(phi)_i^l$

*Loss derivatives* \
$(partial h[theta](x,y)) / (partial w_(i j)^l) = (partial h^l[theta](x^l,y)) / (partial x_i^l) (partial x_i^l) / (partial w_(i j)^l) = delta_i^l dot(phi)_i^l x_j^(l-1)$, \
$(partial h[theta](x,y)) / (partial b_i^l) = (partial h^l[theta](x^l,y)) / (partial x_i^l) (partial x_i^l) / (partial b_i^l) = delta_i^l dot(phi)_i^l$ \
with $delta_i^l = (partial h) / (partial x_i^l) dot(phi)_i^l$

=== Gradient Descent

*Gradient descent update* \
$theta_(t+1) = theta_t - eta nabla h(theta_t)$

*Gradient flow ODE* \
$d theta / dif t = -nabla h(theta)$

*L-smoothness* \
$norm(nabla h(theta_1) - nabla h(theta_2)) <= L norm(theta_1 - theta_2)$ (forall $theta_1, theta_2$) \
$lambda_max(nabla^2 h) <= L$ \
$ell(w) - ell(w') <= nabla ell(w')^top (w - w') + L/2 norm(w - w')^2$ \
$ell''(x) <= L$

*Polyak-Lojasiewicz condition* \
$1/2 norm(nabla h(theta))^2 >= mu (h(theta) - min h)$ (forall $theta$)

*Convergence rate* \
$eta = 1/L$ \
$Delta t = (2 L)/epsilon^2 (h(theta_0) - min h)$ for $epsilon$-critical point \
$Delta h(theta_t) - min h <= (1 - mu/L)^t (h(theta_0) - min h)$

=== Acceleration and Adaptivity

*Heavy ball momentum update* \
$theta_(t+1) = theta_t - eta nabla h(theta_t) + beta (theta_t - theta_(t-1))$

*Nesterov acceleration* \
$tilde(theta)_(t+1) = theta_t + beta (theta_t - theta_(t-1))$ \
$theta_(t+1) = tilde(theta)_(t+1) - eta nabla h(tilde(theta)_(t+1))$ \
More theoretical grounding than heavy ball

*AdaGrad updates* \
$theta_(i,t+1) = theta_(i,t) - eta_i^t (partial h) / (partial theta_i)(theta_t)$, \
$gamma_i^t = gamma_i^(t-1) + ((partial h) / (partial theta_i)(theta_t))^2$, \
$eta_i^t = eta / sqrt(gamma_i^t + delta)$

*Adam updates* \
$g_i^t = beta g_i^(t-1) + (1 - beta) (partial h) / (partial theta_i)(theta_t)$ \
$gamma_i^t = alpha gamma_i^(t-1) + (1 - alpha) ((partial h) / (partial theta_i)(theta_t))^2$ \
$theta_(i,t+1) = theta_(i,t) - eta_i^t g_i^t$, $eta_i^t := eta / sqrt(gamma_i^t + delta)$

*RMSprop* \
Adam without momentum term

=== Stochastic Gradient Descent

*Stochastic gradient descent update* \
$theta_(t+1) = theta_t - eta nabla h(theta_t)(x_(i_t), y_(i_t))$

*SGD variance* \
$V[theta](S) = 1/s sum_(i=1)^s norm(nabla h[theta](S) - nabla h[theta](x_i, y_i))^2$

*SGD convergence rate* \
$E[h(bar(theta)_t)] - min h <= O(1/sqrt(t))$ (general) \
$E[h(bar(theta)_t)] - min h <= O(log t / t)$ (strongly convex) \
$E[h(bar(theta)_t)] - min h <= O(1/t)$ (additionally smooth)

=== Function Properties

*Convexity* \
$ell(lambda w + (1 - lambda) w') <= lambda ell(w) + (1 - lambda) ell(w')$ \
$ell''(x) >= 0$ forall $x$

*Convexity and differentiability* \
$ell(w) >= ell(w') + nabla ell(w')^top (w - w')$ \
Implies convexity for differentiable functions and vice versa

*Strong convexity and differentiability* \
$ell(w) >= ell(w') + nabla ell(w')^top (w - w') + mu/2 norm(w - w')^2$ \
$ell''(x) >= mu$ forall $x$

== Convolutional Networks

=== Convolutions

*Convolution definition* \
$(f * g)(u) := integral_(-oo)^oo g(u-t) f(t) dif t = integral_(-oo)^oo f(u-t) g(t) dif t$

*Fourier transform convolution property* \
$F(f * g) = F(f) * F(g)$

*Discrete convolution* \
$(f * g)[u] := sum_(t=-oo)^oo f[t] g[u - t]$

*Cross-correlation* \
$(g star f)[u] := sum_(t=-oo)^oo g[t] f[u + t]$

*Toeplitz matrices* \
$(f * g) = "Toeplitz-Matrix"(g) f$

=== Convolutional Networks

*Conventions* \
Padding: Add zeros around input \
Stride: Step size of convolution

*Max-Pooling* \
Take maximum value in windows (size $r$)

*ConvNets for Images* \
$y[r][s, t] = sum_u sum_(Delta s, Delta t) w[r, u][Delta s, Delta t] * x[u][s + Delta s, t + Delta t]$ \
$r$: output channel, $u$: input channel

*Number of parameters of a convolutional layer* \
$D = (|r| * |u|) * (|Delta s| * |Delta t|)$ \
fully connected · window size

=== Natural Language Processing with ConvNets

*Word embedding* \
$Omega: w mapsto x_w in RR^n$

*Conditional log-bilinear model* \
Prediction of output word $mu$ given word $w$ in neighborhood \
$P(mu | w) = exp(x_w^top y_mu) / sum_mu exp(x_w^top y_mu)$

$h({x_w}, {y_mu}) = sum_((w,mu)) ell_(w mu)$ \
$ell_(w, mu) = -x_w^top y_mu + ln sum_mu exp(x_w^top y_mu)$

*Negative sampling* \
$tilde(ell)_(w, mu) = -ln sigma(x_w^top y_mu) - beta E_(mu ~ D) ln (1 - sigma(x_w^top y_mu))$

== Recurrent Networks

=== Simple Recurrent Networks

*Time evolution equation* \
$z_t := F[theta](z_(t-1), x_t)$, $z_0 := 0$ (forall $t$)

*Output map* \
$hat(y)_t := G[xi](z_t)$

*RNN parameterization* \
$F[U, V](z, x) := phi(U z + V x)$ \
$G[W](z) := psi(W z)$, $W in RR^(q times m)$

*Backpropagation through time* \
$(partial h) / (partial z_i^t) = sum_(s=t)^T delta_k^s sum_(j=1)^m (partial hat(y)_k^s) / (partial z_j^s) (partial z_j^s) / (partial z_i^t)$, \
$(partial hat(y)_k^s) / (partial z_j^s) = dot(psi)_k^s w_(k j)$ \
$(partial h) / (partial v_(i j)) = sum_(t=1)^T (partial h) / (partial z_i^t) dot(phi)_i^t x_j^t$ \
$(partial h) / (partial u_(i j)) = sum_(t=1)^T (partial h) / (partial z_i^t) dot(phi)_i^t z_j^(t-1)$

*Spectral norm* \
$norm(A)_2 = max_(x: norm(x)=1) norm(A x)_2 = sigma_1(A)$

*Gradient norms* \
$(partial z^T) / (partial z^0) = dot(Phi)^T U * ... * dot(Phi)^1 U$ \
The norm of gradients either: \
1. Vanishes exponentially if $sigma_1(U) < 1 / bar(alpha)$: $norm((partial z^t) / (partial z^0))_2 <= (bar(alpha) sigma_1(U))^t -> oo$ \
2. Explodes if $sigma_1(U)$ is too large

*Bidirectional RNNs* \
$hat(y)_t = psi(W z_t + tilde(W) tilde(z)_t)$

=== Gated Memory

*LSTM* \
$z_t := sigma(F tilde(x)_t) * z_(t-1) + sigma(G tilde(x)_t) * tanh(V tilde(x)_t)$ \
$tilde(x)_t := "mat"(x_t; h_t)$, $h_(t+1) = sigma(H tilde(x)_t) * tanh(U z_t)$

*GRU* \
$z_t = (1 - sigma) * z_(t-1) + sigma * tilde(z)_t$, \
$sigma := sigma(G[x_t, z_(t-1)])$ \
$tilde(z)_t := tanh(V[r_t * z_(t-1), x_t])$ \
$r_t := sigma(H[z_(t-1), x_t])$

=== Linear Recurrent Models

*Linear state evolution* \
$z_(t+1) = A z_t + B x_t$

*Diagonal form* \
$A = P Lambda P^(-1)$, $Lambda := "diag"(lambda_1, ..., lambda_m)$, $lambda_i in CC$

*Stability condition* \
$max_j |lambda_j| <= 1$

*Initialization* \
$lambda_i = exp(-exp(kappa_i) + i phi_i)$, \
$e^(kappa_i) = -ln r_i$ \
$phi_i ~ "Uni"[0; 2 pi]$, $r_i ~ "Uni"[I]$, $I subset [0; 1]$

*Advantages* \
(i) clear modeling of long/short range dependencies \
(ii) no channel mixing required \
(iii) parallelizable training

== Attention and Transformers

=== Attention

*Attention mixing* \
$xi_s := sum_t alpha_(s t) W x_t$, $alpha_(s t) >= 0$, $sum_t alpha_(s t) = 1$ \
$A = (a_(s t)) in RR^(T times T)$, s.t. $Xi = W X A^top$

*Query-key matching* \
$Q = U_Q X$, $K = U_K X$ \
($U_Q, U_K in RR^(q times n)$) \
$Q^top K = X^top U_Q^top U_K X$ rank $<= q$ \
($Q^top K in RR^(T times T)$)

*Softmax attention* \
$A = "softmax"(beta Q^top K)$, \
$a_(s t) = e^(beta [Q^top K]_(s t)) / sum_r e^(beta [Q^top K]_(s r))$ \
usually $beta = 1/sqrt(q)$

*Feature transformation* \
$X mapsto Xi mapsto F(Xi)$, \
$F(theta)(Xi) = (F(xi_1), ..., F(xi_T))$

*Positional encoding* \
$p_(t k) = "cases"(sin(t omega_k), k "even"; cos(t omega_k), k "odd")$, \
$omega_k = C^(k/K)$

*Transformer architecture* \
Self-attention: attend to its own values in the past \
Cross-attention: E.g. decoder attends to encoder output (query from decoder, key and value from encoder)

*Vision transformer patch embedding* \
$RR^(p times p times q) ∋ "patch"_t mapsto x_t := V "vec"("patch"_t) in RR^n$ \
with $V in RR^(n times (q p^2))$

*GELU activation* \
$phi(z) = z "Prob"(z <= Z)$, $Z ~ "N"(0, 1)$

== Geometric Deep Learning

=== Sets and Points

*Function over sets* \
${x_1, ..., x_M} subset RR$, $f: 2^RR -> Y$

*Order-invariance property* \
$f(x_1, ..., x_M) = f(x_(pi(1)), ..., x_(pi(M)))$ forall $pi in S_M$

*Equivariance property* \
$f(x_1, ..., x_M) = (y_1, ..., y_M) => f(x_(pi(1)), ..., x_(pi(M))) = (y_(pi(1)), ..., y_(pi(M)))$

*Permutation invariant sum* \
$sum_(m=1)^M x_m = sum_(m=1)^M x_(pi(m))$, forall $M$, forall $pi in S_M$

*Deep Sets model* \
$f(x_1, ..., x_M) = rho(sum_(m=1)^M phi(x_m))$

*Max pooling variant* \
$f(x_1, ..., x_M) = rho(max_(m=1)^M phi(x_m))$

*Equivariant map construction* \
$rho: RR times RR^N -> Y$, \
$(x_m, sum_(k=1)^M phi(x_k)) mapsto y_m$

=== Graph Convolutional Networks

*Feature and adjacency matrices* \
$X = "mat"(x_1^top; ...; x_M^top)$, $A = (a_(n m))$ \
with $a_(n m) = "cases"(1, "if" {v_n, v_m} in E; 0, "otherwise")$

*Permutation matrix constraints* \
$P in {0, 1}^(M times M)$ s.t. \
$sum_(n=1)^M p_(n m) = sum_(n=1)^M p_(m n) = 1$ (forall $m$)

*Graph invariance definition* \
$f(X, A) != f(P X, P A P^top)$, forall $P in Pi_M$

*Graph equivariance definition* \
$f(X, A) != P f(P X, P A P^top)$, forall $P in Pi_M$

*Node neighborhood features* \
$X_m := {{x_n : {v_n, v_m} in E}}$, ${{"cdot"}} = "multiset"$

*Message passing scheme* \
$phi(x_m, X_m) = phi(x_m, m_(X_m) psi(x))$ \
$m$ is a permutation-invariant operation

*Normalized adjacency matrix* \
$bar(A) = D^(-1/2) (A + I) D^(-1/2)$ \
$D = "diag"(d_1, ..., d_M)$, $d_m = 1 + sum_(n=1)^M a_(n m)$

*GCN layer* \
$X^+ = sigma(bar(A) X W)$, $W in RR^(M times N)$

*Two-layer GCN* \
$Y = "softmax"(bar(A) (bar(A) X W^0) W^1)$

=== Spectral Graph Theory

*Laplacian operator* \
$Delta f := sum_(n=1)^N (partial^2 f) / (partial x_n^2)$, $f: RR^N -> RR$

*Graph Laplacian* \
$L = D - A$, $(L x)_n = sum_(m=1)^M a_(n m) (x_n - x_m)$

*Normalized Laplacian* \
$tilde(L) = I - D^(-1/2) A D^(-1/2) = D^(-1/2) (D - A) D^(-1/2)$

*Graph Fourier transform* \
$L = D - A = U Lambda U^top$, \
$Lambda := "diag"(lambda_1, ..., lambda_M)$, $lambda_i >= lambda_(i+1)$

*Convolution* \
$x * y = U ((U^top x) "odot" (U^top y))$

*Filtering operation* \
$G_theta(L) x = U G_theta(Lambda) U^top x$

*Polynomial kernels* \
$U (sum_(k=0)^K alpha_k Lambda^k) U^top = sum_(k=0)^K alpha_k L^k$

*Polynomial kernel network layer* \
$x_i^(l+1) = sum_j p_(i j)(L) x_j^l + b_i$, \
$p_(i j)(L) = sum_(k=0)^K alpha_(i j k) L^k$

=== Attention GNNs

*Attention coupling matrix* \
$Q = (q_(i j))$, \
$q_(i j) = "softmax"(rho(u^top (V x_i; V x_j; x_(i j))))$ \
s.t. $sum_j A_(i j) q_(i j) = 1$

*Attention propagation* \
$X^+ = sigma(Q X W)$

*Weisfeiler-Lehman test*

== Tricks of the Trade

=== Initialization

*Random initialization* \
$theta_i^0 ~ "N"(0, sigma_i^2)$, or \
$theta_i^0 ~ "Uniform"(-sqrt(3) sigma_i; sqrt(3) sigma_i)$

*LeCun initialization* \
$w_(i j)~ "Uniform"[-a; a]$, $a := 1/sqrt(n)$, $b_i = 0$ \
Stabilizes variance

*Glorot initialization* \
$w_(i j)~ "Uniform"[-sqrt(3) gamma; sqrt(3) gamma]$, \
$gamma := 2/(n + m)$ \
Stabilizes variance of gradients in backpropagation

*He initialization* \
$w_(i j) ~ "N"(0, gamma)$ or $w_(i j) ~ "Uniform"[-sqrt(3) gamma; sqrt(3) gamma]$, \
$gamma := 2/n$ \
In ReLU networks typically only $n/2$ units active

*Orthogonal initialization* \
$1/sqrt(m) W ~ "Uniform"(O(m))$ \
s.t. $W^top W = W W^top = m I$

=== Weight Decay

*L2 regularization* \
$Omega_mu(theta) = mu/2 norm(theta)^2$, $mu >= 0$

*Gradient descent with weight decay* \
$Delta theta = -eta nabla E(theta) - eta nabla Omega_mu(theta) = -eta nabla E(theta) - eta mu theta$

*Weight decay for multiple layers* \
$theta = ("vec"(W^1), "vec"(W^2), ..., "vec"(W^L))$, \
$Omega_mu(theta) = sum_(l=1)^L mu_l norm(W^l)_F^2$

*Local loss landscape* \
$theta_mu^"*" = (H + mu I)^(-1) H theta^"*"$, $H = Q^top Lambda Q$ \
$(Lambda + I)^(-1) Lambda = "diag"(lambda_i / (lambda_i + mu))$ \
The minimum $theta^"*"$ is shrunk along directions with small eigenvalues

*Generalization* \
$mu = sigma^2 / u^2$, $u$: teacher "sign"al \
Optimal weight decay inverse proportional to the "sign"al-to-noise ratio

=== Dropout

*Probability $phi_i$ of keeping a unit*

*Dropout as Ensembling* \
$p(y | x) = sum_(b in {0,1}^R) p(b) p(y | x; b)$ \
with $p(b) = "prod"_(i=1)^R phi_i^(b_i) (1 - phi_i)^(1 - b_i)$

*Weight scaling for inference* \
$tilde(w)_(i j) <- phi_j w_(i j)$

=== Normalization

*Batch normalization* \
$E$ and $V$ from minibatches or population statistics \
$bar(f) = (f - E[f]) / sqrt(V[f])$, $E[bar(f)] = 0$, $V[bar(f)] = 1$ \
$bar(f)[mu, gamma] = mu + gamma bar(f)$

*Weight normalization* \
$f(v, gamma)(x) = phi(w^top x)$, $w := gamma / norm(v)_2 v$ \
Gradient descent with respect to decoupled $gamma$ and $v$: \
$(partial E) / (partial gamma) = nabla_w E * v / norm(v)_2$ \
$nabla_v E = gamma / norm(v) (I - (w w^top) / norm(w)^2) nabla_w E$

*Layer normalization* \
$tilde(f)_i = (f_i - E[f]) / sqrt(V[f])$, \
$E[f] = 1/m sum_(i=1)^m f_i$ \
$V[f] = 1/m sum_(i=1)^m (f_i - E[f])^2$ \
Using population averages across units in a layer

=== Model Distillation

*Tempered cross entropy loss for distillation* \
$ell(x) = sum_(y=1)^K (exp[F_y(x)/T]) / (sum_(mu=1)^K exp[F_mu(x)/T]) [1/T G_y(x) - ln sum_(mu=1)^K exp[G_mu(x)/T]]$ \
$T > 0$, $F_y$: teacher logits, $G_y$: student logits

*Gradient of distillation loss* \
$(partial ell) / (partial G_y) = 1/T [e^(F_y/T) / sum_mu e^(F_mu/T) - e^(G_y/T) / sum_mu e^(G_mu/T)]$

== Theory

=== Neural Tangent Kernel

*Linearized DNN taylor approximation* \
$h(beta)(x) = f(x) + beta * nabla f(x)$ \
with $beta approx theta - theta_0$, $f(x) := f(theta_0)(x)$

*Kernel of gradient feature maps* \
$k(x, xi) = nabla f(x) * nabla f(xi)$, $RR^d times RR^d -> RR$

*Dual representation* \
$h(alpha)(x) = f(x) + sum_(i=1)^s alpha_i nabla f(x_i) * nabla f(x)$

*Squared loss* \
$E(alpha) = 1/(2s) sum_(i=1)^s (sum_(j=1)^s alpha_j nabla f(x_j) * nabla f(x_i) + f(x_i) - y_i)^2$

*Optimal solution of linearized DNN* \
$K = [k(x_i, x_j)]_(i,j=1)^n in RR^(n times n)$ \
$alpha^"*" = K^+ (y - f)$, \
$h^"*"(x) = k(x) K^+ (y - f)$

*Neural Tangent Kernel NTK* \
$k(theta)(x, xi) := nabla f(theta)(x) * nabla f(theta)(xi)$

*Quadratic loss* \
$E(theta) = 1/2 norm(f(theta) - y)^2$, $y := (y_1, ..., y_s)^top$

*Gradient flow ODE* \
$dot(theta) := dif theta / dif t = sum_(i=1)^s (y_i - f_i(theta)) nabla f_i(theta)$

*Functional gradient flow* \
$dot(f)_j = nabla f_j * dot(theta) = sum_(i=1)^s (y_i - f_i) k(theta)(x_i, x_j)$ \
$dot(f) = K(theta)(y - f)$

*Infinite width limit* \
$w_(i j)^l = sigma_w / sqrt(m_l) epsilon_(i j)^l$, \
$b_i^l = sigma_b / sqrt(m_l) beta_i^l$, \
$epsilon_(i j)^l, beta_i^l ~ "N"(0, 1)$ \
$k(theta) -> k_oo$ for $m_l -> oo$ \
Initial NTK converges to deterministic limit

*NTK constancy* \
$d k(theta(t)) / dif t = 0$ \
$f_oo(x) = k(x) K^+ (y - f)$, $k = k_oo$ \
NTK remains constant when training in infinite width limit

*Vanishing curvature* \
$norm(nabla^2 f(theta_0))_2 / norm(nabla f(theta_0))_2^2 << 1$

*Near-constancy* \
$norm(k(theta_0) - k(theta_t))_F in O(1/m)$, $m = m_1 = ... = m_L$

=== Bayesian DNNs

*Bayesian predictive distribution* \
$f(x) = integral f(theta)(x) p(theta | S) d theta$

*Bayes rule* \
$p(theta | S) = (p(theta) p(S | theta)) / p(S)$, \
$p(S) = integral p(theta) p(S | theta) d theta$

*Parameter priors (Gaussian)* \
$p(theta) = "prod"_(i=1)^d p(theta_i)$, $theta_i ~ "N"(0, sigma_i^2)$ \
$-log p(theta) = 1/(2 sigma^2) norm(theta)^2 + "const"$ \
Essentially a weight decay term

*Likelihood (Gaussian noise)* \
$-log p(S | theta) = 1/(2 gamma^2) norm(y - f(theta))^2 + "const".$ \
with $y_i = f^"*"(x_i) + nu_i$, $nu_i ~ "N"(0, gamma^2)$

*Posterior* \
$-log p(theta | S) = E(theta) + "const"$, \
$E(theta) = 1/(2 gamma^2) norm(y - f)^2 + 1/(2 sigma^2) norm(theta)^2$

*Bayesian ensembling (post hoc)* \
$f(Theta)(x) = sum_(i=1)^n (exp[-E(theta_i)]) / (sum_(j=1)^n exp[-E(theta_j)]) f(theta_i)(x)$ \
Relative posterior weighting

*Markov chain monte carlo (MCMC)* \
$theta_0, theta_1, theta_2, ...$, \
$theta_(t+1) | theta_t ~ Pi$ \
$p(theta_1 | S) Pi(theta_2 | theta_1) = p(theta_2 | S) Pi(theta_1 | theta_2)$

*Metropolis-Hastings* \
$Pi(theta_1 | theta_2) = tilde(Pi)(theta_1 | theta_2) A(theta_1 | theta_2)$ \
$A(theta_1 | theta_2) = min{1, (p(theta_1 | S) tilde(Pi)(theta_2 | theta_1)) / (p(theta_2 | S) tilde(Pi)(theta_1 | theta_2))}$ \
Modified transition probability with acceptance step $A$

*Hamiltonian monte carlo* \
$E(theta) = -sum_(x,y) log p(y | x; theta) - log p(theta)$ \
$H(theta, v) = E(theta) + 1/2 v^top M^(-1) v$ \
with $p(theta, v) "propto" exp[-H(theta, v)]$ \
$dot(v) = -E(theta)$, $dot(theta) = v$ \
$theta_(t+1) = theta_t + eta v_t$ \
$v_(t+1) = v_t - eta nabla E(theta_t)$

*Langevin dynamics* \
$dot(theta) = v$ \
$d v = -nabla E(theta) dif t - B v dif t + N(0, 2B dif t)$ \
$theta_(t+1) = theta_t + eta v_t$ \
$v_(t+1) = (1 - eta gamma) v_t - eta integral nabla tilde(E)(theta) + sqrt(2 gamma eta) N(0, I)$

=== Gaussian Processes

*Gaussian process* \
$(f(x_1), ..., f(x_s)) ~ N$ \
$sum_(i=1)^s alpha_i f(x_i) ~ N$, forall $alpha in RR^s$

*Mean and covariance functions* \
GPs are completely defined by first and second order statistics \
$mu(x) := E_x[f(x)]$ \
$k(x, xi) := E_(x,xi)[f(x) f(xi)] - mu(x) mu(xi)$ \
$K_(mu nu) = k(x_mu, x_nu)$, $K in RR^(s times s)$

*Example kernels* \
$k(x, xi) = x^top xi$, $k(x, xi) = e^(-gamma norm(x - xi)^2)$

*GPs in DNN* \
Treating parameters as random variables. Each unit in a DNN becomes a random function.

*Linear Layer* \
$w ~ "N"(0, sigma^2/n I_(n times n))$ \
$E[y_i y_j] = sigma^2/n x_i^top x_j$

*Deep layers* \
$W^(l+1) X^l$, $l >= 1$ \
No longer normal as products break normality, but near-normal for high dimensional inputs.

*Non-linear activations* \
$mu(x^(l+1)) = E[phi(W^l x^l)]$ \
*Kernel recursion* \
$K_(mu nu)^l = E[phi(x_(i mu)^(l-1)) phi(x_(i nu)^(l-1))]$ \
$= sigma^2 E[phi(f_mu) phi(f_nu)]$ \
$f~ "GP"(0, K^(l-1))$

*Kernel regression* \
Mean of bayesian predictive distribution \
$f^"*"(x) = k(x)^top K^+ y$ \
$E[(f(x) - f^"*"(x))^2] = K(x, x) - k(x)^top K^+ k(x)$

=== Statistical Learning Theory

*VC learning theory* \
$L_t = -norm(m(x_t, x_0, t) - m_theta(x_t, t))^2 / (2 sigma_t^2) + "const".$ \
$"VC-dim"(F) := max_s sup_(|S|=s) 1[|F(S)| = 2^s]$

*VC inequality* \
$P(sup_F |hat(E)(f) - E(f)| > epsilon) <= 8 |F(s)| e^(-s epsilon^2 / 32)$

*Double descent* \
Beyond the interpolation point, models start to learn and eventually may level out at a lower generalization error.

*Generalization gap* \
$Delta := max(0, E - hat(E))$ \
$E$: expected population error, $hat(E)$: empirical error

*KL divergence* \
$D_("KL")(p || q) = integral p(x) log (p(x) / q(x)) dif x = E_(x ~ p)[ln (p(x) / q(x))]$

*PAC-Bayesian theorem* \
For fixed $E$ and any $Q$ over $s$ samples: \
$E_Q[E(f)] - E_Q[hat(E)(f)] <= sqrt(2/s ["KL"(Q || P) + ln (1 / (2 sqrt(s) epsilon))])$ \
Ensures general rate $tilde(O)(1/sqrt(s))$

*PAC-Bayesian bound* \
$Q := N(theta, "diag"(sigma_i^2))$ \
$"KL"(Q || P) = sum_i log (lambda / sigma_i + (sigma_i^2 + theta_i^2) / (2 lambda^2) - 1/2)$ \
$E_("PAC")(Q) := E_Q[hat(E)] + sqrt(2/s ["KL"(Q || P) + ln (1 / (2 sqrt(s) epsilon))])$ \
Favours minima robust to parameter perturbations

*PAC-bayesian learning implementation* \
$theta_(t+1) = theta_t - eta nabla E_Q[hat(E)] = theta_t - eta nabla hat(E)(tilde(theta))$, \
with $tilde(theta) ~ Q(theta, sigma)$ \
Gradient loss on perturbed parameters

*Reparameterization trick* \
$tilde(theta) = theta + "diag"(sigma_i) epsilon$, $epsilon ~ "N"(0, I)$ \
Backpropagation to $theta$ and $sigma_i$

== Generative Models

=== Variational Autoencoders

*Linear autoencoder* \
$x mapsto z = C x$, $C in RR^(m times n)$ \
$z mapsto hat(x) = D z$, $D in RR^(n times m)$ \
$E(C, D)(x) = 1/2 norm(x - hat(x))^2 = 1/2 norm(x - D C x)^2$ \
$D C X = hat(X) = U Sigma_m V^top$ \
$Sigma_m = "diag"(sigma_1, ..., sigma_m, 0, ..., 0)$ \
For centered data equivalent to PCA, but generally has non-global minima

*Linear factor analysis* \
*Probability Model* \
$p_X(x) = integral p_Z(z) p_(X|Z)(x | z) dif z$ \
$Z$: latent variables, $X$: observed variables

*Linear observation model* \
$x = mu + W z + nu$ with $nu ~ "N"(0, Sigma)$ \
$x ~ "N"(mu, W W^top + Sigma)$ for $z ~ "N"(0, I)$

*Posterior mean and covariance* \
$mu_(z|x) = W^top (W W^top + Sigma)^(-1) (x - mu)$ \
$Sigma_(z|x) = I - W^top (W W^top + Sigma)^(-1) W$

*Pseudoinverse limit* \
$W^top (W W^top + sigma^2 I)^(-1) -> W^+ in RR^(m times n)$ \
$mu_(z|x) -> W^+(x - mu)$, $Sigma_(z|x) -> 0$

*Maximum likelihood estimation* \
$mu, W max-> log p_(mu, W)(S)$

*Optimality condition for W* \
$w_i = rho_i u_i$, $rho_i = max{0, sqrt(lambda_i - sigma^2)}$ \
With $(lambda_i, u_i)$ eigenvalues and eigenvectors of covariance matrix. \
For $sigma = 0$ equivalent to PCA.

*Variational autoencoder (VAE)* \
$z ~ "N"(0, I)$ \
$x = F(theta)(z) = (F^L @ ... @ F^1)(z)$

*Evidence lower bound (ELBO)* \
$log p_(theta)(x) = log integral p_(theta)(x | z) p(z) dif z$ \
$= log integral q(z) [(p_(theta)(x | z) p(z)) / q(z)] dif z$ \
$>= integral q(z) log p_(theta)(x | z) dif z - integral q(z) log (q(z) / p(z)) dif z$ \
$=: L(theta, q)(x)$ \
$theta max-> L(theta, q)(S) = sum_(i=1)^s L(theta, q)(x_i)$

*Inference network* \
$z ~ "N"(mu(x), Sigma(x))$ \
$z = mu + Sigma^(1/2) epsilon$, $epsilon ~ "N"(0, I)$ \
$nabla_mu E[f(z)] = E[nabla_z f(z)]$ \
$nabla_Sigma E[f(z)] = 1/2 E[nabla_z^2 f(z)]$ \
Integration by parts derivation

=== Generative Adversarial Networks

*GAN objective* \
$V(G, D) = E_(x_r ~ p_"data") D(x_r) + E_(z ~ p_z)(1 - D(G(z)))$

*Discriminator Mixture Model* \
$tilde(p)_theta(x, y) = 1/2 (y p(x) + (1 - y) p_theta(x))$, \
$y in {0, 1}$, \
$p$: true probability, $p_theta$: model probability

*Bayes-optimal classifier* \
$q_theta(x) := P{y = 1 | x} = p(x) / (p(x) + p_theta(x))$ \
To detect fake samples, $y = 1$ for real samples, $y = 0$ for fake samples

*Logistic likelihood* \
$theta min-> ell^"*"(theta) := E_(tilde(p)_theta)[y ln q_theta(x) + (1 - y) ln(1 - q_theta(x))]$

*Jensen-Shannon as effective objective* \
$ell^"*" = E_(tilde(p)_theta)[y ln q_theta(x) + (1 - y) ln(1 - q_theta(x))]$ \
$= -1/2 H(p) - 1/2 H(p_theta) + H(1/2 (p + p_theta)) - ln 2$ \
$= "JS"(p, p_theta) - ln 2.$

*Discriminator model* \
$q_phi: x mapsto [0; 1]$, $phi in Phi$

*Objective bounds* \
$ell^"*"(theta) >= sup_(phi in Phi) ell(theta, phi)$ \
$ell(theta, phi) := E_(tilde(p)_theta)[y ln q_phi(x) + (1 - y) ln(1 - q_phi(x))]$

*Saddle point optimization* \
$theta^"*" := "argmin"_(theta in Theta) (sup_(phi in Phi) ell(theta, phi))$ \
$phi$: Generator, $theta$: Discriminator

*Alternating gradient descent/ascent* \
$theta_(t+1) = theta_t - eta nabla_theta ell(theta_t, phi_t)$ \
$phi_(t+1) = phi_t + eta nabla_phi ell(theta_(t+1), phi_t)$

*Extra-gradient steps* \
$theta_(t+1) = theta_t - eta nabla_theta ell(theta_(t+0.5), phi_t)$ \
with $theta_(t+0.5) := theta_t - eta nabla_theta ell(theta_t, phi_t)$ \
$phi_(t+1) = phi_t + eta nabla_phi ell(theta_t, phi_(t+0.5))$ \
with $phi_(t+0.5) := phi_t + eta nabla_phi ell(theta_t, phi_t)$

*Deconvolutional DNN* \
Upside-down ConvNet for image generation

=== Denoising Diffusion

*Markov chains* \
$x_(0:t-1) perp x_(t+1:oo) | x_t$ (forall $t$) \
$p(x_t | x_(t-1)) = p(x_1 | x_0)$ (forall $t$), \
$p(x_(s:t)) = p(x_t) "prod"_(tau=s+1)^t p(x_(tau-1) | x_tau)$ \
$p(x_(s:t)) = p(x_s) "prod"_(tau=s+1)^t p(x_tau | x_(tau-1))$, \
$pi(x_(t+1)) = integral pi(x_t) p(x_(t+1) | x_t) dif x_t$

*Denoising diffusion* \
*Forward (noise generation)* \
$pi^"*" = nu_0 mapsto nu_1 mapsto ... mapsto nu_(T-1) mapsto nu_T = pi$ \
*Backward (denoising)* \
$pi = mu_T^theta mapsto mu_(T-1)^theta mapsto ... mapsto mu_1^theta mapsto mu_0^theta approx pi^"*"$

*Gaussian example* \
$pi approx N(0, I)$, \
$x_t | x_(t-1) ~ "N"(sqrt(1 - beta_t) x_(t-1), beta_t I)$ \
*Forward SDE* \
$d x_t = -1/2 beta_t x_t dif t + sqrt(beta_t) dif omega_t$ \
*Backward SDE* \
$d x_t = [-1/2 beta_t x_t - beta_t nabla_(x_t) log q_t(x_t)] dif t + sqrt(beta_t) dif bar(omega)_t$ \
score · wiener process

*ELBO bound* \
$x_t = sqrt(1 - beta_t) x_(t-1) + sqrt(beta_t) epsilon_t$, $epsilon_t ~ "N"(0, I)$ \
$ln p_theta(x_0) = ln integral q(x_(1:T) | x_0) (p_theta(x_(0:T)) / q(x_(1:T) | x_0)) dif x_(1:T)$ \
$>= E[ln (p_theta(x_(0:T)) / q(x_(1:T) | x_0)) | x_0]$ \
$= sum_(t=0)^T L_t$ \
$L_t := "cases"(E[ln p_theta(x_0 | x_1)], t = 0; -D(q(x_T | x_0) || pi), t = T; -D(q(x_(t-1) | x_t, x_0) || p_theta(x_(t-1) | x_t)), "else")$

*Backward model assumption* \
$x_(t-1) | x_t ~ "N"(m(x_t, t), Sigma(x_t, t))$

*Entropy bounds* \
$H(x_t) >= H(x_(t-1)) => H(x_t | x_(t-1)) >= H(x_(t-1) | x_t)$

*Noise schedules* \
$bar(alpha)_t = "prod"_(tau=1)^t (1 - beta_tau)$, $bar(beta)_t = 1 - bar(alpha)_t$ \
$x_t approx N(sqrt(bar(alpha)_t) x_0, bar(beta)_t I)$ $t->oo->$ $N(0, I)$

*Forward trajectory target* \
$x_(t-1) | x_t, x_0 = N(m(x_t, x_0, t), tilde(beta)_t I)$ \
$m(x_t, x_0, t) = (sqrt(bar(alpha)_(t-1) beta_t) / (1 - bar(alpha)_t)) x_0 + ((1 - bar(alpha)_(t-1)) sqrt(1 - beta_t)) / (1 - bar(alpha)_t) x_t$ \
with $tilde(beta)_t = (1 - bar(alpha)_(t-1)) / (1 - bar(alpha)_t) beta_t$

*Fixed isotropic covariance* \
$Sigma(x_t, t) = sigma_t^2 I$, where $sigma_t^2 in {beta_t, tilde(beta)_t}$

*Simplified ELBO* \
$L_t = -norm(m(x_t, x_0, t) - m_theta(x_t, t))^2 / (2 sigma_t^2) + "const".$

*Reparameterization* \
$x_t = sqrt(bar(alpha)_t) x_0 + sqrt(1 - bar(alpha)_t) epsilon => x_0 = 1/sqrt(bar(alpha)_t) x_t - sqrt(1 - bar(alpha)_t) / sqrt(bar(alpha)_t) epsilon$ \
$m(x_t, x_0, t) = 1/sqrt(alpha_t) [x_t(x_0, epsilon) - beta_t / sqrt(1 - bar(alpha)_t) epsilon]$ \
with $epsilon ~ "N"(0, I)$

*Expected squared error* \
$E_q[L_t | x_0] = E_epsilon[rho_t norm(epsilon - epsilon_theta(sqrt(bar(alpha)_t) x_0 + sqrt(1 - bar(alpha)_t) epsilon, t))^2 | x_0]$ \
with $rho_t = beta_t^2 / (2 sigma_t^2 alpha_t (1 - bar(alpha)_t))$

*Final simplified criterion* \
$h(theta)(x) = 1/T sum_(t=1)^T E[norm(epsilon - epsilon_theta(sqrt(bar(alpha)_t) x + sqrt(1 - bar(alpha)_t) epsilon, t))^2]$

== Ethics

=== Adversarial Examples

*Adversarial perturbation* \
$f(x + nu) != f(x)$ s.t. $norm(nu)_p <= epsilon$

*p-norm definitions* \
$norm(x)_p = (sum_i |x_i|^p)^(1/p)$ \
$norm(x)_oo = max_i |x_i|$, $norm(x)_0 = |{i : x_i != 0}|$

*Optimal perturbation (linear binary classification)* \
$nu prop "sign"(f_1(x) - f_2(x)) (w_2 - w_1)$ \
for $f_i = w_i^top x + b_i$

*Optimal perturbation (multiclass)* \
$nu = "argmin"_(i > 1) (f_1(x) - f_i(x)) / norm(w_1 - w_i)_2^2 (w_i - w_1)$

*DeepFool iterative optimization* \
Iterate: $"argmin"_(Delta nu) norm(Delta nu)_2$ s.t. \
$(nabla f_1(x) - nabla f_2(x))^top Delta nu < f_1(x) - f_2(x)$

*Robust training* \
$ell(f(x), y) -> max_(nu: norm(nu)_p <= epsilon) ell(f(x + nu), y)$

*Projected gradient ascent (p = 2)* \
$nu_(t+1) = epsilon Pi[nu_t + alpha nabla_x ell(f(x + nu_t), y)]$ \
$Pi[z] := z / norm(z)_2$

*Projected gradient ascent (p = oo)* \
$nu_(t+1) = epsilon Pi[nu_t + alpha "sign"(nabla_x ell(f(x + nu_t), y))]$ \
$Pi[z] := z / norm(z)_oo$

*Fast Gradient Sign Method (FGSM)* \
$nu = epsilon "sign"(nabla_x ell(f(x), y))$



#v(1000pt)
#h(1pt)
#v(1000pt)
#h(1pt)
#v(1000pt)
#h(1pt)
#v(1000pt)

= Computer Vision

== The Digital Image & Sensors

#colorbox(title: [Charge Coupled Device (CCD)], inline:false, color:gray)[
Photons 
  - *Blooming*: Oversaturated photosites cause vertical channels to "flood" (bright vertical line)
]


#colorbox(title: [Image Noise], inline:false)[
  Additive Gaussian noise:
]

*Color camera concepts:*
1. Prism (split light, 3 sensors, needs good alignment, good color separation)

== Image Segmentation
Pixel-wise classification problem, to group pixels in an image that share common properties. \
Segmentation of $I$: Find $R_1, ..., R_n$ such that \
$I = union.big_(i = 1)^N R_i$ with $R_i inter R_j = emptyset quad forall i != j$.

#colorbox(title: [Thresholding], color: silver, inline: true)[
  Segment image into 2 classes. \
  $B(x, y) = 1 "if" I(x, y) >= T "else" 0$, finding $T$ with trial and error, compare results with ground truth.
]

#colorbox(title: "Important Kernels", color: purple, inline: true)[
  
  #set math.mat(delim: "[")
  #v(-7pt)
  #grid(columns: (auto, auto, auto, auto), gutter: 1em,
    [Laplacian], [$"Prewitt"_x$], [#v(-10pt) Low-pass/ \ Mean / Box], [High-pass], 
    [Gaussian], [$"Sobel"_x$], [$"Diff"_x$], [$"Diff"_y$],
    [\ $ 1 / (2 pi sigma^2) e^(-(x^2 + y^2) / (2 sigma^2))$],
    $"mat"(-1,0,1; -2,0,2; -1,0,1)$,
    $"mat"([-1], 1)$,
    $"mat"([-1], 1)^top$
  )

]

*Dirac delta*: $delta(x) = "cases"(0 "if" x!=0, "undefined else")$ with $integral_(-oo)^infinity delta(x) dif x = 1$. $cal(F)[delta(x - x_0)](u) = e^(-i 2 pi u x_0)$. $delta(u) = integral_RR e^(-i 2 pi x u) dif x$.\
*Sampling* $f$ at points $x_n$: $f_("s")(x)=sum_(n)f(x_n)delta(x-x_n)$.
#v(-3pt)
#grid(columns: (auto, auto, auto), column-gutter: 1.5em, row-gutter: 0.3em,
  [*Property*], $bold(f(x))$, $bold(F(u))$,
  [Linearity], $alpha f_1(x) + beta f_2(x)$, $alpha F_1(u) + beta F_2(u)$,
  [Duality], $F(x)$, $f(-u)$
)

#grid(columns: (60%, 39%), column-gutter: 0.4em, image("fourier-transforms.png", height: 16.6em), [*Simple procedure of sampling and reconstructing a 2D signal*: Sample Signal, $"FT"$, Cut out Magnitude Spectrum by multiplication with box filter, $"FT"^(-1)$. 
*Some reconstruction filters*: Nearest neighbor, Bilinear ,])
#v(-14pt)
Gaussian reconstruction filter (equiv. to convolving sampled signal w/ Gaussian kernel. $sinc(x)=sin(pi x)/(pi x)$


#set align(left)

#colorbox(title: [Image restoration], inline: false)[
Image degradation is applying kernel $h$ to some image $I$. The inverse $tilde(h)$ should compensate: 
$I xarrow(sym: -->, h(x)) J xarrow(sym: -->, tilde(h)(x)) I$. \
Determine with $cal(F)[tilde(h)](u, v) dot.c cal(F)[h](u, v) = 1$. Or $tilde(h) = cal(F)^(-1) [1/(cal(F)^[h])]$ \
Cancellation of frequencies & noise amplification $->$ Regularize using $tilde(cal(F))[tilde(h)](u, v) = cal(F)[h] slash.big (|cal(F)[h]|^2 + epsilon)$.
]
*Motion blur*: $h(x,y) = 1/(2l)[theta (x+l) - theta (x-l)] delta (y)$