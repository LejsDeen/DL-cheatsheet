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
$f[w, b](x) = "sign"(x dot  w + b)$ with *Decision Boundary* $x dot w + b =^! 0$. $-b/norm(w)$ is the signed distance of the hyperplane from $0$. \
*Geometric Margin* $gamma[w, b](x, y) = (y (x dot  w + b)) / norm(w)$.
*Maximum Margin Classifier*\
$(w^"*", b^"*") in "argmax"_(w,b) gamma[w, b](cal(S))$,
with margin $gamma[w, b](cal(S)) := min_((x,y) in cal(S)) gamma[w, b](x, y)$.

#colorbox(title: [Perception Learning], inline:false)[
  `if` $f[w, b](x) != y$:  $w<- w+y x$, $b <-b +y$. \
  Aims to find some solution; assumes version space is non-empty. Does not aim for small err.
  #v(-6pt)
  #fitWidth($w_0 in "span"(x_1, ..., x_s) => w_t in "span"(x_1, ..., x_s) forall t$)
]

*Convergence - Novikoff's Theorem* \
1. #text(0.76em)[If $exists w^*$, $norm(w^*) = 1$, s.t. $gamma[w^*](cal(S)) = gamma > 0 => w_t  dot  w^* >= t gamma$.]
2. Let $R = max_(x in S) norm(x)$. Then $ norm(w_t) <= R sqrt(t)$. \
#v(-6pt)
#text(0.83em)[$cos angle(w^*, w_t) = (w^*  dot  w_t) / (norm(w^*) norm(w_t)) >= (t gamma) / (sqrt(t) R) = sqrt(t) gamma / R <=^! 1 => t <= R^2 / gamma^2$.]

#colorbox(title: [Cover's Theorem for $cal(S) subset RR^n, |cal(S)| = s$],color: silver, inline:false)[
  $C(cal(S), n)$: \# of ways to separate $cal(S)$ in $n$ dimensions. Position of pts does not matter (as long as they are in general positition).
  $C(s + 1, n) = 2 sum_(i=0)^(n-1) binom(s, i)$, 
  $C(s, n) = 2^s$ for $s <= n$. 
  Phase transition at $s = 2n$. For $s < 2n$ empty version space is the exception, otherwise the rule.
]

== Hopfield Networks


$E(X) = -1/2 sum_(i != j) w_(i j) X_i X_j + sum_i b_i X_i$,
where $X_i in {plus.minus 1}$. $w_(i j) = w_(j i)$ , $w_(i i) = 0$.

#colorbox(title: [Hebbian Learning], color: silver, inline:false)[
  Choose patterns ${x^t}^s_(t=1) in {plus.minus 1}^n$, compute weights once using them:
$w_(i j) = 1/n sum_(t=1)^s x_i^t x_j^t$, $w_(i i) = 0$. For inference, update $X$ iteratively: $X_i^(t+1) = "sign"(sum_j w_(i j) X_j^t + b_i)$ asynchronously.
Capacity for random, uncorrelated patterns: $s_"max" approx 0.138 n$. Requiring pattern to be retrieved with high probability: $s <= n/(2 log_2 n)$.
]

#colorbox(color:purple)[
  If $X="diag"(1, ...,1)$, no reconstruction happens. \
  Under async update step, any Hopfield network is guaranteed to converge.
]


= Feedforward Networks

== Linear Models

*Linear regression* (MSE)\
$L[w](X, y) =  norm(X w - y)^2/(2n)$,
$nabla_w L = (X^top X w - X^top y) / n$.
*Moore-Penrose inverse solution* \
$w^* = X^* y in "argmin"_w L[w](X, y)$,
where $X^* = lim_(delta -> 0) (X^top X + delta I)^(-1) X^top$ Moore-Penrose inverse.

*Stochastic gradient descent update* 
$w_(t+1) = w_t + eta (y_(i_t) - w_t^top x_(i_t)) x_(i_t)$, $i_t ~ cal(U)([1, n])$.

*Gaussian noise model* 
$y_i = w^top x_i + epsilon_i$, $epsilon_i ~ cal(N)(0, sigma^2)$, LSQ equivalent to NLL of gaussian noise model. 

*Ridge regression* 
$h_lambda [w] = h[w] + lambda/2 norm(w)^2$,
$w^"*" = (X^top X + lambda I)^(-1) X^top y$.


*Logistic function* \
$sigma(z) = 1/(1 + e^(-z))$, $sigma(z) + sigma(-z) = 1$. \
$sigma' = sigma(1 - sigma)$, $sigma'' = sigma(1 - sigma)(1 - 2 sigma)$ \
*Cross entropy loss* for $y in {0, 1}$\
$ell(y, z) = -y log sigma(z) - (1 - y) log(1 - sigma(z))$ \
$= -log sigma((2y - 1)z)$. \
*Logistic regression with Cross Entropy loss*:
$L[w]=1/n sum_(i=1)^n ell_i (y_i , w^top x_i) $,$nabla ell_i = [sigma(w^top x_i) - y_i] x_i$.

== Feedforward Networks

*Generic feedforward layers* \
#text(0.7em)[$F: underbrace(RR^(m(n+1)), "parameters") times underbrace(RR^n, "input") -> underbrace(RR^m, "output")$], $F[theta](x)  = phi(W x + b)$. \
*Layer composition*
$G = F^L [theta^L] circle.small dots circle.small F^1[theta^1]$. \
*Layer activations* 
$x^l = F^l circle.small dots circle.small F^1(x) .= F^l (x^(l-1))$, $x^0 = x$, $x^L = F(x)$.

*Softmax*$(z)_i = e^(z_i) / (sum_j e^(z_j))$,
CE-loss in terms of logits $ell(y, z) = 1/ln(2) [-z_y + ln sum_j e^(z_j)]$.
CE between two pmfs: $l(p;q) = -sum_i p_i log q_i$.\
CE with hard labels is NLL-loss.

*Residual layer* $F[W, b](x) = x + (phi(W x + b) - phi(0))$,
therefore $F[0, 0] = id$. Link that propagates $x$ forward is called a *skip connection*. Composing residual layers: number of paths grows exponentially, can include projections for flexiblity of changing dimensionality.

== Sigmoid Networks

*Sigmoid activation* $sigma(z) = 1/(1 + e^(-z))$. \
*Hyperbolic tangent activation* \
$tanh(z) = (e^z - e^(-z))/(e^z + e^(-z)) = 2 sigma(2z) - 1$. \
$tanh'(z) = 1 - tanh^2(z)$.

*Smooth function approximation* \
Polynomials, ridge functions ($phi(a^top x + b)$) and MLPs with $C^oo$ activations are universal approximators. \
*Weierstrass:* Polynomials are universal approximators of $C(RR)$ on any given compact $I$. \
*Barron's Theorem:*
For $f: RR^d arrow RR$ with $C_f = integral norm(omega) |hat(f)(omega)| dif omega < oo$, $exists$ width-$m$ MLP $g_m$ s.t.: $integral_B |f-g_m|^2 dif x <= O(1/m)$.

== ReLU$(z)=max(0, z)$ networks
*Zalavsky's Thoerem: Activation patterns* \
$m$ ReLU neurons in $RR^n$. Each neuron's hyperplane ${w_i ^top x= 0}$ partitions $RR^n$ into $R(m)$ connected regions of constant activation pattern. $R(m) <= sum_(i=0)^(min(n,m)) binom(m, i) << 2^m$. \
*Montufar: Connected linear regions in ReLU network* 
$R(m, L) >= R(m) floor(m/n)^(n(L-1))$, $L$: layers, $m$: width. \
*Shektman*: Piecewise linear functions are dense in $C([0;1])$. 
*Lebesgue*: Piecewise linear function with $m$ pieces can be written $g(x) = a x + b + sum_(i=1)^(m-1) c_i (x-x_i)_+$; $m+1$ parameters, $a, b, c_i$. \
*ReLU networks with 1 hidden layer are universal approximators.* \
*Wang and Sun*: Every continuous piecewise linear function $g: RR^n -> RR$ can be written as a signed sum of $k$-Hinges with $k <= n+1$. A $k$-Hinge is a function $g(x) = max_(j=1)^k {w_j^top x + b_j}$, generalizes ReLU, known as Maxout unit.

*Linear Autoencoder*: Optimal $A=D E$, s.t. frobenius norm reconstruction err of $A X$ is minimized, is $D=U_k, E=U_k ^top$, not jointly convex in $E$ and $D$, but individually. $hat(X)^* = arg min_("rank"(hat(X))=k)$$norm(X - hat(X))^2_F = U Sigma_k V^top$ SVD. 

= Gradient-Based Learning

Forward mode is more memory efficient, but backward mode is more runtime efficient. Fwd is $O("#params")$, reverse is $O(d_"out")$. \
*Numerator layout*: 
For $f : RR^n -> RR^m$,
$((partial y)/(partial x))_(i j) = (partial y_i)/(partial x_j) in RR^(m times n)$ and $f : RR^(n_1 times n_2) -> RR$, $nabla f(X)_(i j) = (partial f)/(partial X_(i j)) in RR^(n_1 times n_2)$.

== Backpropagation

$x^ell = phi(W^ell x^(ell-1) + b^ell)$,
$(partial cal(L))/(partial W^ell) = delta^ell (x^(ell-1))^top$, 
$(partial cal(L))/(partial b^ell) = delta^ell$, 
$delta^ell = (partial cal(L))/(partial x^ell) dot.circle phi'(W^ell x^(ell-1) + b^ell), (partial cal(L))/(partial x^ell)=(W^(ell+1))^top delta^(ell+1)$.

#colorbox(color: purple)[
$(partial L) / (partial x) = (partial L) / (partial z) (partial z) / (partial x)$. For $x in RR^n$ different $z$: 
$(partial (W x))/(partial x) = W$, element wise $f$ gives : $(partial f(x))/(partial x) = "diag"(f'(x))$, $(partial norm(hat(y)-y)^2)/(partial hat(y)) = 2 (hat(y) - y)^top$, $(partial L)/(partial hat(y)) (partial (W h))/(partial W) = h dot (partial L) / (partial hat(y))$.

$d/(d x_j) "softmax"(x)_i = "sm"(x)_i (delta_(i j) - "sm"(x)_j)$]

== Gradient Descent

*Update*: $x_(t+1) = x_t - eta nabla f(x_t)$.

*Gradient flow ODE* $(dif x) / (dif t) = -nabla f(x)$ gives ideal trajectory to be approximated by gradient descent.

*Newton's method* gives optimal step for quadratic model: $Delta x = - [nabla^2 f(x)]^(-1) nabla f(x)$. \
$nabla^2_x [x^top A x + b^top x + c] = A + A^top $

*Optimal LR for Convex Quadratics* \
For $f(x) = 1/2 x^top Q x$, $eta^star = 2/(lambda_max (Q) + lambda_min (Q))$. Stability requires $eta <= 2/(lambda_max (Q))$. Quadratic approx. of $f$: $f(x + Delta x) approx f(x) + nabla f(x)^top Delta x + 1/2 Delta x^top nabla^2 f(x) Delta x$. Condition number of $Q$: $lambda_"max"/lambda_"min"$

*L-smooth:* $norm(nabla f(x) - nabla f(y)) <= L norm(x - y)$. \
Equivalently (if $f$ twice diff) (for $L=0 =>$): \ 
$f(y) <= f(x) + nabla f(x)^top (y-x) + L/2 norm(y-x)^2$. \
Implies $lambda_i (nabla^2 f(x)) <= L$ for all EVs $lambda_i$ of $nabla^2 f(x)$. \
*Convexity* ($lambda in [0,1]$): \ $f(lambda w + (1 - lambda) w') <= lambda f(w) + (1 - lambda) f(w')$. \
*$mu$-Strong convexity:* ($mu = 0 <=>$ convex + diff)\
$<=>f(y) >= f(x) + nabla f(x)^top (y-x) + mu/2 norm(y-x)^2$. \
Implies $lambda_i (nabla^2 f(x)) >= mu$ for all EVs $lambda_i$ of $nabla^2 f(x)$. \
For $f: RR -> RR$, these become: $L >= f''(x) >= mu quad forall x$.
#colorbox[
If $f$ $mu$-strongly convex, $L$-smooth, GD iterates $x_t$ with $0<eta <= 1/L$ converge to unique minimizer $x^*$ at rate $norm(x_t - x^*)^2 <= (1-eta mu)^k norm(x_0 - x^*)^2$. \
If $f$ convex, diff and $L$-smooth, with $eta <= 1/L$, $f(x_t) - f(x^*) <= 1/(2 eta t) norm(x_0 - x^*)^2$.
*Non-Convex case*: If $f$ diff, $L$-smooth, with minimum $f^*$, GD iterates with $eta <=1/L$ satisfy
$min_(i=0)^t norm(nabla f(x_i))^2 <= (2(f(x_0)-f^*))/(eta(t +1))$.
] 

If $f$ diff and $L$-smooth: $f(x) - f(x^*) >= 1/(2L) norm(nabla f(x))^2$.

*Polyak-Lojasiewicz condition* \
$1/2 norm(nabla f(x))^2 >= mu (f(x) - min f)$ (forall $x$). \
$mu$-strong convex $=>$ $mu$-PL.

#colorbox(title: [GD Convergence Rates & Learning Rates], color: blue, inline:false)[
  *L-smooth only:* $eta^star = 1/L$. To reach $epsilon$-stationary point ($norm(nabla f) <= epsilon$) needs at most $(2L)/epsilon^2 (f(x_0) - min f)$ steps. \
*$mu$-PL + L-smooth:* Use $eta^* = 2/(L+mu)$. Convergence: \ $f(x_t) - f(x^*) <= (1-mu/L)^t (f(x_0) - f(x^*))$.
]



== Stochastic Gradient Descent

*SGD variance* \
#text(size: 0.9em)[
$V[theta](S) = 1/s sum_(i=1)^s norm(nabla f[theta](S) - nabla f[theta](x_i, y_i))^2$]. 

Polyak averages: $overline(x)_(k+1) = k/(k+1) overline(x)_k + 1/(k+1) x_(k+1)$

*SGD convergence rate* with Polyak averaging and $eta_k prop 1/k$ \
#text(size: 0.9em)[
$EE[f(overline(theta)_t)] - min f <= O(1/sqrt(t))$ (general) \
$EE[f(overline(theta)_t)] - min f <= O((log t) / t)$ (strongly convex) \
$EE[f(overline(theta)_t)] - min f <= O(1/t)$ (additionally smooth)

*Minibatch SGD:* Variance $arrow.b$ by $prop r$. Can $arrow.t$ $ eta prop r$.]

*Var. Reduction with SVRG* w/ occasional snapshot $overline(theta)$: $theta_(t+1) = theta_t - eta [nabla f_i (theta_t) - nabla f_i (overline(theta)) + nabla f(overline(theta))]$.

== Acceleration and Adaptivity

*Heavy ball momentum update* \
$theta_(t+1) = theta_t - eta nabla f(theta_t) + beta (theta_t - theta_(t-1))$

*Nesterov acceleration* \
$tilde(theta)_(t+1) = theta_t + beta (theta_t - theta_(t-1))$ \
$theta_(t+1) = tilde(theta)_(t+1) - eta nabla h(tilde(theta)_(t+1))$ \
More theoretical grounding than heavy ball.

*AdaGrad updates* \
$theta_(t+1) = theta_(t) -  eta / sqrt(gamma_t + epsilon) dot.circle nabla f(theta_t)$, \
$gamma^t = gamma^(t-1) + nabla f(theta_t) dot.o nabla f(theta_t)$.

*Adam updates* \
$m_t = beta_1 m_(t-1) + (1-beta_1) nabla f(theta_t)$, $hat(m)_t = m_t / (1-beta_1^t)$\
$v_t = beta_2 v_(t-1) + (1-beta_2) (nabla f(theta_t))^2$, $hat(v)_t = v_t / (1-beta_2^t)$\
$theta_(t+1) = theta_t - eta hat(m)_t / (sqrt(hat(v)_t) + epsilon)$. \
*RMSprop*: Adam without momentum term.

*signSGD*: $theta_(t+1) = theta_t - eta "sign"(nabla f_(I_k)(theta_t))$.

#colorbox(color: purple)[
$x^4$ is strictly convex but not strongly convex, since near $0$ the growth of $x^4$ is slower than $x^2$, violating the uniform lower bound on curvature. \
With $f$ $L$-smooth and $mu$-PL, GD with optimal step-size $arg min_eta f(theta_t - eta nabla f(theta_t))$ converges globally at linear rate. \
$f(w)=(norm(X w - y)^2)/2 + lambda norm(w)^2$ satisfies PL-condition. \
*Muon*: Orthogonalize gradient, should increase the scale of other rare directions which have small magnitude in update but are important. \
$Delta W = - norm(nabla L(W))_* dot d_"out"/d_"in" dot U^top V$ ($nabla L(W) = U Sigma V^top$) minimizes RHS of: $L(W + Delta W) <= L(W) + chevron nabla_W L(W), Delta W chevron.r_F + 1/2 d_"in"/d_"out" norm(Delta W)^2_2$. \
GD Trajectory always orthogonal to level set.]

= Convolutional Networks

Convolution $(f*g)(u) = (g*f)(u)$

$= integral_(-oo)^oo f(t)g(u-t) dif t = integral_(-oo)^oo f(u-t) g(t) dif t$. \
$(f*g) = "Toeplitz-Matrix"(g) f$.

*Fourier transform convolution property* \
$cal(F)(f * g) = cal(F)(f) dot cal(F)(g)$.

*Cross-correlation*: \
$(g star f)[u] = sum_t g[t] f[u + t]$.

#colorbox(color:purple)[
$(f(t) star g(t))(y) = (f(-t) * g(t))(y)$. \
$(f(t)*g(t))(-y) = (f(-t)*g(t))(y) = (f(t)*g(-t))(y)$. \
Translation-Equivariant Operators $=$ Convolutions. \
For $x in RR^D, h in RR^K$, $x * h = W_h x.$
*Toeplitz Matrix*: $W_h in RR^((D-K+1) times D)$\ $(W_h)_(k,j) = cases(
  h_(K+k-j) & "if" k <= j <= k+K-1,
  0 & "otherwise"
)$ for $h in RR^K$.

$nabla_w v^top "vec"(sigma (x * w)) = "flip"[x] * ["mat"(v) dot.o sigma' (x * w)]$, flip rows and columns.

Normal convolution of image $x in RR^(d times d)$, kernel $w in RR^(q times q)$ $x*w$ requires $cal(O)((d-q)^2 q^2)$. If $w$ separable st $w= u v^top$, $u in RR^q, v in RR^p$, $cal(O)(d q (d-q))$

Output size: $H_"out" = floor((H_"in" + 2 P - K)/S) + 1$, Height, Input size, padding, Kernel size, stride.
]

== Convolutional Networks

*Conventions* for Padding: Add zeros around input.

*ConvNets for Images* ($r$ out channel, $u$ in channel)\
$y[r][s, t] = sum_u sum_(i, j) w[r, u][i, j] * x[u][s + i, t + j]$.

*Number of parameters of a convolutional layer* \
$(|r| times |u|) dot (|i| times |j|)$ : fully connected $times$ patchsize.

== Word2Vec
Per word $omega$, have input embedding $x_omega$ and output embedding $y_omega$.

Predict context word $nu$ given center word $omega$:
$P(nu | omega) = exp(x_omega^top y_nu) / (sum_mu exp(x_omega^top y_mu))$.

NLL loss:
$ell_(omega,nu) = -x_omega^top y_nu + ln sum_mu exp(x_omega^top y_mu)$.
Total: $h({x_omega}, {y_nu}) = sum_((omega,nu)) ell_(omega,nu)$ over observed pairs.
Use only input embeddings after training.

= Geometric Deep Learning

*Group* is set $G$ with a binary operation s.t.: 1) $(g h) f = g (h f)$, 2) $exists e in G$ s.t. $e f = f e = f$, 3) $forall g space exists g^(-1) in G$ s.t. $g g^(-1) = g^(-1) g = e$, 4) $g h in G space forall g, h$. *Abelian* if $g h = h g$.

== Sets and Points

*Order-invariance property:* \
$f(x_1, ..., x_M) = f(x_(pi(1)), ..., x_(pi(M)))$ (perturbations).

*(Permutation) Equivariance property:* \
$f(x_1, ..., x_M) = (y_1, ..., y_M) => f(x_(pi(1)), ..., x_(pi(M))) = (y_(pi(1)), ..., y_(pi(M)))$

*Deep Sets model* (invariant layer): \
$f(x_1, ..., x_M) = rho(sum_(m=1)^M phi(x_m))$.

*Equivariant map construction:* \
$rho: RR times RR^N -> Y$,
$(x_m, sum_(k=1)^M phi(x_k)) mapsto y_m$

== Graph Convolutional Networks

*Feature and adjacency matrices* \
$X = "mat"(x_1^top; ...; x_M^top)$, $A = (a_(n m))$ \
with $a_(n m) = 1 <=> {v_n, v_m} in E$.

*Permutation matrix constraints* \
$P in {0, 1}^(M times M)$ with single 1 in each row and col.

*Graph invariance definition* \
$f(X, A) =^! f(P X, P A P^top)$, $forall P in Pi_M$.

*Graph equivariance definition* \
$f(X, A) =^! P f(P X, P A P^top)$, $forall P in Pi_M$.

*Node neighborhood features* \
$X_m = {{x_n : {v_n, v_m} in E}}$, ${{dot.c}} = "multiset"$

*Message passing scheme* \
$phi(x_m, X_m) = phi(x_m, plus.o.big_(x in X_m) psi(x))$, \
$plus.o$ is some permutation-invariant operation.

*Normalized adjacency matrix* \
$overline(A) = D^(-1/2) (A + I) D^(-1/2)$, \
$D = "diag"(d_1, ..., d_M)$, $d_m = 1 + sum_(n=1)^M a_(n m)$.

*One GCN layer* \
$X^+ = sigma(overline(A) X W)$, $W in RR^(M times N)$.

=== Spectral Graph Theory

*Laplacian operator* \
$Delta f = sum_(n=1)^N (partial^2 f) / (partial x_n^2)$, $f: RR^N -> RR$.

*Graph Laplacian* \
$L = D - A$, $(L x)_n = sum_(m=1)^M a_(n m) (x_n - x_m)$. \
$x^top L x = 1/2 sum_u sum_v A_(u v) (x_u - x_v)^2 >= 0$ (psd).

*Normalized Laplacian* \
$tilde(L) = I - D^(-1/2) A D^(-1/2) = D^(-1/2) (D - A) D^(-1/2)$.

*Graph Fourier transform* \
$L = D - A = U Lambda U^top$, $hat(f) = U^top f$, $f =  U hat(f)$. \
$Lambda := "diag"(lambda_1, ..., lambda_M)$, $lambda_i >= lambda_(i+1)$.

*Convolution:* 
$x * y = U ((U^top x) dot.o (U^top y))$.

*Filtering operation* \
$G_theta (L) x = U G_theta (Lambda) U^top x$

*Polynomial kernels* \
$U (sum_(k=0)^K alpha_k Lambda^k) U^top = sum_(k=0)^K alpha_k L^k$

*Polynomial kernel network layer* \
$x_i^(l+1) = sum_j p_(i j)(L) x_j^l + b_i$, \
$p_(i j)(L) = sum_(k=0)^K alpha_(i j k) L^k$

#colorbox(color:purple)[
  GNNs cannot distinguish between certain graphs that are topologically different. Unconstrained set architectures are more powerful.
If *WL-test* says graphs are different, they are; but if it says they're the same, they might still be different.
]

= Theory of DNNs

== Statistical Learning Theory

*Risk decomposition* \
$f^*$ : optimal predictor over all functions, \ $f_H^* = "argmin"_(f in H) cal(R)(f)$, $hat(f)_H$:  learned from finite data. \
#text(0.8em)[
$underbrace(cal(R)(hat(f)_H)-cal(R)(f^*), "excess risk") = underbrace(cal(R)(hat(f)_H) - cal(R)(f^*_H), "estimation error") + underbrace(cal(R)(f^*_H) - cal(R)(f^*), "approximation error")$.
]

*Rademacher Complexity* $G={g_h | h in H}$: \
For $sigma in {-1, 1}$, measures how well $G$ can fit random noise:
$hat(frak(R))_(D_n)(G) = EE_sigma [sup_(g in G) 1/n sum_(i=1)^n sigma_i g(z_i)]$. \
$EE["sup"_(h in H)cal(R)(h) - hat(cal(R))_(D_n)(h)] <= 2 hat(frak(R))_(D_n)(G)$, \
$EE[cal(R)(hat(h)_H)] <= cal(R)(h^*_H) + 2 frak(R)_(D_n)(G)$.

*Double descent*: \
Beyond the interpolation point, models eventually may level out at a lower generalization error.

*Implicit bias towards min norm solutions*: Any convergent algorithm with iterates in $"span"{x_1, ..., x_n}$ finds the minimum norm solution.

=== A PAC-Bayesian result

$P$ prior distribution over functions before seeing data, $Q$ posterior after training.

*PAC-Bayesian theorem* \
Bounds generalization gap for stochastic classifiers ($f ~ Q$): 
$E_Q [cal(R)(f)] - E_Q [hat(cal(R))_n (f)] <= sqrt(2/n [op("KL")(Q||P) + ln (2 sqrt(n) \/ epsilon)])$ \
- $P$: prior, $Q$: posterior (learned). Rate $tilde(O)(1\/sqrt(n))$ \
- $op("KL")(Q||P)$: "information cost" of moving $P -> Q$ \
- Insight: generalization depends on *distance moved*, not parameter count

*PAC-Bayesian for DNNs* \
$P = cal(N)(0, lambda I)$, $Q = cal(N)(theta, op("diag")(sigma_i^2))$ \
$op("KL")(Q||P) = sum_i [ log lambda / sigma_i + (sigma_i^2 + theta_i^2) / (2 lambda^2) - 1/2 ]$ \
Minimize directly: $E_Q [hat(cal(R))] + sqrt(2/n [op("KL")(Q||P) + ...])$ \
$=>$ encourages wide/flat minima (perturbations $theta + epsilon$ must also perform well)

*Implementation:* Reparameterization: $tilde(theta) = theta + op("diag")(sigma_i) eta$, $eta ~ cal(N)(0, I)$,
Backprop through $theta$ and $sigma$.

== Linearized DNNs and NTK

Training neural network $f(bold(theta))(x)$ can be approximated by *linearizing* around initialization $bold(theta)_0$ when parameters change slowly.


*Linearization $->$ Kernel Regression:* \
Taylor approximation: $h(bold(beta))(x) = f(bold(theta)_0)(x) + bold(beta) dot.op nabla f(bold(theta)_0)(x), quad bold(beta) = bold(theta) - bold(theta)_0$. \
With residuals $tilde(y)_i = y_i - f(bold(theta)_0)(x_i)$, training becomes *linear regression* with features $nabla f(bold(theta)_0)(x_i)$: $min norm(tilde(y)_i - beta dot nabla f(theta_0))$

*Neural Tangent Kernel (NTK):* \
Definition: $k_(bold(theta))(x, x') := nabla f(bold(theta))(x) dot.op nabla f(bold(theta))(x')$.

*Dual representation:* \
$h(bold(alpha))(x) = f(bold(theta)_0)(x) + sum_(i=1)^n alpha_i k_(bold(theta)_0)(x_i, x)$. \
*Optimization problem:* $min_(bold(alpha)) 1/2 norm(bold(K)_(bold(theta)_0) bold(alpha) - tilde(bold(y)))^2$ \
*Optimal solution (kernel regression):* \
$bold(alpha)^* = bold(K)_(bold(theta)_0)^dagger (bold(y) - bold(f)(bold(theta)_0))$,
$h^*(x) = bold(k)_(bold(theta)_0)(x)^top bold(alpha)^*$


*Functional Gradient Flow* \
Training dynamics in function space: \
$dot(bold(f))(bold(theta)) = bold(K)(bold(theta))(bold(y) - bold(f)(bold(theta)))$
- If $bold(K)(bold(theta))$ constant $→$ *linear ODE* with closed-form solution
- If $bold(K)(bold(theta))$ evolves $→$ *nonlinear dynamics*, feature learning

*Infinite-Width Limit* \
Initialization: $w_(i j)^((ell)) tilde (sigma_w)/sqrt(m_ell) cal(N)(0,1)$.
Result: As width $m -> infinity$:
$k_(bold(theta)(t)) -> k_infinity$ (constant during training).
- Kernel becomes *deterministic* (depends only on architecture/init scheme)
- Training = kernel regression with frozen $k_infinity$
- No feature learning
*Finite-width:* $norm(bold(K)(bold(theta)_0) - bold(K)(bold(theta)(t))) = cal(O)(1/m)$

*Why Kernel Stays Constant:* \
Kernel grad $nabla K = nabla^2 f(x) nabla f(z) + nabla^2 f(z) nabla f(x)$, \
at $m = infinity$: $nabla^2 f -> 0 => nabla K -> 0$ → kernel frozen.

#table(
  columns: 2,
  stroke: 0.5pt,
  align: left,
  inset: (x: 1pt, y: 4pt),
  [*Lazy Training \ (NTK Regime)*], [*Feature Learning \ (Rich Regime)*],
  [$m -> infinity$, small LR], [Finite width, normal LR],
  [$bold(K)$ const $→$ linear dynamics], [$bold(K)$ evolves $→$ nonlinear],
  [No feature learning], [Learns representations],
  [Theoretically tractable], [SOTA performance]
)

*Takeaways:* \
*Linearization* turns NN training into kernel regression with features $nabla f(bold(theta)_0)(x)$. \
*NTK* $k_(bold(theta)) = nabla f(bold(theta))(x) dot.op nabla f(bold(theta))(x')$ governs training dynamics via $dot(bold(f)) = bold(K)(bold(y) - bold(f))$. \
*Infinite width* $→$ kernel constant $→$ NN = kernel machine (no feature learning).\
*Finite width* $→$ kernel evolves $cal(O)(1/m)$ $→$ enables feature learning.\
*NTK explains lazy regime but NOT why deep learning works* $->$ real power is feature learning when kernel changes.

== Random NNs and GPs

*Marginals and Conditionals of MV Gaussians* \

Let $X in RR^d tilde cal(N)(bold(mu), bold(Sigma))$ with partition: \ 
$X = vec(X_A, X_B), quad 
bold(mu) = vec(bold(mu)_A, bold(mu)_B), quad
bold(Sigma) = mat(bold(Sigma)_(A A), bold(Sigma)_(A B); 
                   bold(Sigma)_(B A), bold(Sigma)_(B B))$

*Marginal:* $X_A tilde cal(N)(bold(mu)_A, bold(Sigma)_(A A))$

*Conditional:* $X_B | X_A tilde cal(N)(bold(mu)_(B|A), bold(Sigma)_(B|A))$

$bold(mu)_(B|A) &= bold(mu)_B + bold(Sigma)_(B A) bold(Sigma)_(A A)^(-1) (X_A - bold(mu)_A) \
bold(Sigma)_(B|A) &= bold(Sigma)_(B B) - bold(Sigma)_(B A) bold(Sigma)_(A A)^(-1) bold(Sigma)_(A B)$

=== Bayesian Linear Regression

*Least-squares:* 
$hat(bold(w)) = arg min_bold(w) 1/(2n sigma^2) norm(bold(y) - bold(X) bold(w))^2$.

*Closed-form solution:*
$hat(bold(w)) = (bold(X)^top bold(X))^(-1) bold(X)^top bold(y)$.

#text(0.95em)[
*MLE interpretation:* $y_i = bold(x)_i^top bold(w) + epsilon_i$, $epsilon_i tilde cal(N)(0, sigma^2)$], 
$cal(L)(bold(w)) = log product_(i=1)^n p(y_i | bold(x)_i, bold(w))$, \
$y_i | bold(x)_i, bold(w) tilde cal(N)(bold(x)_i^top bold(w), sigma^2)$.

*Prior:* $p(bold(w)) = cal(N)(bold(0), bold(I)_d)$. \
*Posterior:* 
$p(bold(w) | bold(y), bold(X)) = (p(bold(y) | bold(X), bold(w)) p(bold(w))) / (p(bold(y) | bold(X)))$.

*Predictive distribution:*
$p(y_(n+1)) = integral p(y_(n+1) | bold(w)) p(bold(w) | bold(y)) dif bold(w)$.

$bold(Sigma)_(w w) = I_d$, $bold(Sigma)_(y w) = bold(X), bold(Sigma)_(y y) = bold(X) bold(X)^top + sigma^2 I_n$.

$bold(mu)_(w|y) = bold(X)^top bold(Sigma)_(y y)^(-1) bold(y) = (bold(X)^top bold(X) + sigma^2 bold(I)_d)^(-1) bold(X)^top bold(y)$, \
$bold(Sigma)_(w|y) = I_d - bold(X)^top bold(Sigma)_(y y)^(-1) bold(X) = sigma^2 (bold(X)^top bold(X) + sigma^2 bold(I)_d)^(-1)$.

Same result as ridge: $hat(bold(w)) = (bold(X)^top bold(X) + sigma^2 bold(I)_d)^(-1) bold(X)^top bold(y)$.

Equiv. to GP with linear kernel: $f(dot) tilde cal(G P)(0, k(dot, dot))$: 
$k(x, x') = phi.alt(x)^top phi.alt(x')$, $y=f+epsilon$, $f tilde cal(N)(0, K)$, $epsilon tilde cal(N)(0, sigma^2 I_n)$, posterior $p(f|y) = cal(N)(bold(mu)_(f|y), bold(Sigma)_(f|y))$, $bold(mu)_(f|y) = bold(K) (bold(K) + sigma^2 I_n)^(-1) bold(y)$.

#colorbox(color: purple)[
For $y = f(x) + epsilon$, $f(x) = x^top w$, $epsilon ~ cal(N)(0, sigma^2)$, $p(y | X, w) = cal(N)(X^top w, sigma^2 I)$. If $w ~ cal(N)(0, Sigma)$, $p(w | x, X) = cal(N)(Sigma_"post" 1/sigma^2 X y, Sigma_"post")$, where $Sigma_"post"^(-1) = Sigma^(-1) + 1/sigma^2 X X^top$. Maximizing $log p(w | x, X)$ is same as minimizing least-squares with $ell_2$ penalty $1/2 w^top Sigma^(-1) w$, ridge when $Sigma = lambda^(-1) I$. *Predictive*: $f_* | x_*, y, X = x_*^top w ~ cal(N)(x_*^top mu_"post", x_*^top Sigma_"post" x_*)$. $y_* | x_*, y, X$ adds $sigma^2$. \
$f = x^top w$ with $w ~cal(N)(0, Sigma)$ is GP w $k(x, x')=x^top Sigma x'$. 
]

=== NNGPs

*Setup:* Random 1-hidden-layer NN with $m$ units: \
$f(bold(x)) = v_0 + 1/sqrt(m) sum_(j=1)^m v_j phi(bold(theta)_j^top bold(x))$.
Random init: $v_0 tilde cal(N)(0, sigma_0^2)$, $EE[v_j^2] = sigma_v^2$, $"Cov"(bold(theta)_j) = bold(Sigma)_theta$. 

*Result:* As $m arrow infinity$, $f(dot.c) arrow "GP"(0, k)$ where \
$k(bold(x), bold(x)') = sigma_0^2 + sigma_v^2 EE_(bold(theta)) [phi(bold(theta)^top bold(x)) phi(bold(theta)^top bold(x)')]$

*Monte Carlo approximation:* Sample $B$ random NNs ${f_b}_(b=1)^B$, define features:
- $phi(bold(x)) = 1/sqrt(B) (f_1(bold(x)), dots, f_B(bold(x)))^top$
- $bold(Phi) = [bold(f)_1 dots bold(f)_B] in RR^(n times B)$ (feature matrix)
- $hat(bold(K)) = bold(Phi) bold(Phi)^top$ (approximate kernel matrix)

*GP regression:* Posterior mean and variance: \
$EE[f(bold(x)) | bold(y)] = phi(bold(x))^top (bold(Phi)^top bold(Phi) + sigma^2 bold(I)_B)^(-1) bold(Phi)^top bold(y)$ \
$"Var"[f(bold(x)) | bold(y)] = sigma^2 phi(bold(x))^top (bold(Phi)^top bold(Phi) + sigma^2 bold(I)_B)^(-1) phi(bold(x))$

*Key advantage:* Inverts $B times B$ matrix instead of $n times n$ when $B << n$.

= Generative Models


=== Linear Autoencoders

*Setup.* Encoder $bold(C) in RR^(k times d)$, decoder $bold(D) in RR^(d times k)$, data $bold(X) in RR^(d times n)$ (centered cols):
$min_(bold(C), bold(D)) norm(bold(X) - bold(D) bold(C) bold(X))_F^2$.

*Optimal Solution (PCA).* Let $bold(S) = bold(X) bold(X)^top$ with eigendecomposition $bold(S) = bold(Q) bold(Lambda)^2 bold(Q)^top$, $lambda_1 >= dots.h >= lambda_d >= 0$.
Optimal reconstruction via rank-$k$ projection:
$ hat(bold(X)) = bold(U)_k^* bold(U)_k^(* top) bold(X)$
where $bold(U)_k^* = bold(Q)_([: , 1:k])$ are top-$k$ eigenvectors of $bold(S)$ (equiv, top-$k$ left singular vectors of $bold(X)$).
- Any $bold(C) = bold(U)_k^(* top) bold(A)$, $bold(D) = bold(A)^(-1) (bold(U)_k^*)^top$ is optimal ($forall A$)
- Reduces to truncated SVD: $hat(bold(X)) = bold(U)^* bold(Lambda)_k bold(V)^top$ with $bold(Lambda)_k = op("Diag")(lambda_1, dots, lambda_k, 0, dots, 0)$
- Convex objective with no spurious local minima (gradient descent finds global optimum)
- Singular vectors may not be uniquely identified

=== Factor analysis

*Latent Variable Models* are a generic way to describe generative models. Latent variable $z tilde p(z)$, conditional models for observables $x, p(x|z)$, observed data model: $p(x) = integral p(x|z) p(z) dif z$.

*Mixture models*: simple discrete models: $z in [K]$, $p(z)$ mixing proportions, $p(x|z)$ condit. densities. 

$x tilde cal(N)(bold(mu), bold(Sigma))$: $p(x;bold(mu), bold(Sigma)) = exp[-1/2 (x - bold(mu))^top bold(Sigma)^(-1) (x - bold(mu))] / sqrt((2 pi)^n "det"(bold(Sigma)))$

=== Linear Factor Analysis

- Latent prior: $bold(z) tilde cal(N)(bold(0), bold(I))$, $bold(z) in RR^m$
- Observation: $bold(x) = bold(mu) + bold(W) bold(z) + bold(eta)$, $bold(eta) tilde cal(N)(bold(0), bold(Sigma))$
- Independence: $bold(eta) perp bold(z)$
- Typically $m < n$ (fewer factors than features)

*Marginal distribution:*
$bold(x) tilde cal(N)(bold(mu), bold(W) bold(W)^top + bold(Sigma))$
- $bold(W) bold(W)^top$: shared variance (low-rank, explained by latent factors)
- $bold(Sigma)$: unique variance (diagonal, observation-specific)

Non-identifiability: $(bold(W) bold(Q))(bold(W) bold(Q))^top = bold(W) bold(Q) bold(Q)^top bold(W)^top$ \ $= bold(W) bold(W)^top$ 
for any orthogonal $bold(Q)$. Factors only identifiable up to rotations/reflections.
$arrow.r.double$ Use factor rotations (varimax, etc.) for interpretability.

*MLE estimation:*
$bold(theta) = (bold(mu), bold(W)) arrow.l.long^max log p(bold(X); bold(mu), bold(W))$
- $hat(bold(mu)) = 1/s sum_(i=1)^s bold(x)_i$ (closed form)
- No closed form for $bold(W)$ $arrow.r$ use GD or EM algorithm

*Posterior (encoder):*
$p(bold(z) | bold(x)) = (p(bold(x) | bold(z)) p(bold(z))) / p(bold(x))$.\
$bold(mu)_(bold(z)|bold(x)) = bold(W)^top (bold(W) bold(W)^top + bold(Sigma))^(-1) (bold(x) - bold(mu))$. \
$bold(Sigma)_(bold(z)|bold(x)) = bold(I) - bold(W)^top (bold(W) bold(W)^top + bold(Sigma))^(-1) bold(W)$.

*Probabilistic PCA:*
Special case $bold(Sigma) = sigma^2 bold(I)$. Optimal $i$-th column:
$bold(w)_i = rho_i bold(u)_i, quad rho_i^2 = max{0, lambda_i - sigma^2}$. $W = U_m L_m$\
where $(lambda_i, bold(u)_i)$ is $i$-th eigenpair of data covariance.

As $sigma arrow 0$: $bold(mu)_(bold(z)|bold(x)) arrow bold(W)^dagger (bold(x) - bold(mu))$ (standard PCA). If $W$ has orthogonal columns, then $W^dagger = W^top$.

== Variational Autoencoders

$bold(z) in RR^d$ is learned embedding of $bold(x)$. For generation,  $bold(z) tilde cal(N)(bold(0), I)$, decoder $p_theta (bold(x)|bold(z))$ maps latent to data.

*Problem:* $p_theta (bold(x)) = integral p(bold(z)) p_theta (bold(x)|bold(z)) dif bold(z)$ intractable. \
*Solution:* Maximize ELBO instead:
$log p_theta (bold(x)) >= underbrace(EE_(q_phi.alt (bold(z)|bold(x)))[log p_theta (bold(x)|bold(z))], "reconstruction") - underbrace(D_"KL" (q_phi.alt (bold(z)|bold(x)) || p(bold(z))), "regularization")$

- *Reconstruction:* Encode $bold(x) ->^(q_phi.alt) bold(z)$, decode back
- *KL term:* Keep encoder output close to prior $p(z) tilde cal(N)(bold(0), I)$, ensures  generation using latents

*Encoder* $q_phi.alt (bold(z)|bold(x)) = cal(N)(bold(mu)_phi.alt (bold(x)), "diag"(bold(sigma)_phi.alt^2 (bold(x))))$.

*KL closed form*: 
$D_"KL" (q_phi.alt (bold(z)|bold(x)) || p(bold(z)))$\
$ = 1/2 sum_(j=1)^d (sigma_(phi.alt, j)^2 + mu_(phi.alt, j)^2 - 1 - log sigma_(phi.alt, j)^2)$.

#colorbox(color:blue)[
  $"KL"(cal(N)(mu_0, sigma_o^2) || cal(N)(mu_1, sigma_1^2)) = 1/2 (sigma_0^2/sigma_1^2 + (mu_0-mu_1)^2/sigma_1^2 - 1 + log sigma_1^2/sigma_0^2)$
]

#colorbox(color:purple)[
  $"KL"(p|| q)) = EE_p [log p(x)/q(x)]$
]

- *Fwd KL*: $q_1^* = arg min_(q in Q) "KL"(p || q)$
- *Rev KL*: $q_2^* = arg min_(q in Q) "KL"(q || p)$
Rev KL: Mode-seeking ($p=0 => q=0$), \
FwdKL: Mean-seeking ($p!=0 => q!=0$). \ MLE minimizes fwd KL to empirical $cal(D)$.

#text(size:0.9em)[$log p_theta (bold(x)|bold(z)) = -1/(2sigma^2) norm(bold(x) - bold(mu)_theta (bold(z)))^2_2 - d/2 log(2 pi sigma^2)$.]

*Reparameterization trick:* $bold(z) = bold(mu)_phi (bold(x)) + bold(sigma)_phi (bold(x)) dot.o bold(epsilon)$, $bold(epsilon) tilde cal(N)(bold(0), I)$, enables backprop through sampling.

$log p_theta (bold(x)) - "ELBO" = D_"KL" (q_phi.alt (bold(z)|bold(x)) || p_theta (bold(z)|bold(x)))$, *tight when $q_phi.alt = $ true posterior*.

*Monte Carlo estimation*: $E_(q_phi.alt (bold(z)|bold(x)))[log p_theta (bold(x)|bold(z))] approx -1/(2sigma^2 K) sum_(k=1)^K norm(bold(x) - bold(mu)_theta (bold(z_k)))^2_2 - d/2 log(2 pi sigma^2)$.

#colorbox(color:purple)[
  *Generative Classifiers*
  Given $y in {0,1}$, $p(y=1)=p(y=0)=1/2$, $p(x | y) = cal(N)(x; mu_y, I_d)$, where $mu_0, mu_1 in RR^d$, $p(y=1 | x) = (p(y=1)  p(x | y=1))/(p(y=1)  p(x | y=1) + p(y=0)  p(x | y=0)) = (1/2 (2 pi)^(-d/2) exp(-1/2 norm(x-mu_1)^2))/(1/2 (2 pi)^(-d/2) exp(-1/2 norm(x-mu_1)^2) + 1/2 (2 pi)^(-d/2) exp(-1/2 norm(x-mu_0)^2)) = 1/(1+ exp(1/2 norm(x-mu_1)^2 - 1/2 norm(x-mu_0)^2)) = 
  1/(1+ exp(-[(mu_1 - mu_0)^top x + 1/2 (norm(mu_0)^2 - norm(mu_1)^2)])$ equiv to logistic regression where $p(y=1|x)=sigma(w^top x + b)$ with $w = mu_1 - mu_0, b=1/2 (norm(mu_0)^2 - norm(mu_1)^2)$.

  *ELBO for Hierarchical VAEs*: Model $x$ by decoding from latents $z=(z_1, dots, z_L)$. 
  $p_theta (x,z) = p_theta (x|z_1) product_(i=1)^(L-1) p_theta (z_i|z_(i+1)) p(z_L)$. 
  *Inference* top-down: $q_phi.alt (z|x) = q_phi.alt (z_L|x) product_(i=1)^(L-1) q_phi.alt (z_i | z_(i+1))$. \
  *ELBO* for HVAE: $cal(L)(x) = EE_(z|x ~ q_phi.alt) [log (p_theta (x,z))/(q_phi.alt (z|x))] = 
  EE_(z|x ~ q_phi.alt) [log p_theta (x|z_1) + log (p_theta (z_1|z_2))/(q_phi.alt (z_1|x)) + sum_(i=2)^(L-1) log (p_theta (z_i|z_(i+1)))/(q_phi.alt (z_i|z_(i-1))) + log (p_theta (z_L))/(q_phi.alt (z_L|z_(L-1)))]$.
  
  *Change of variables* spherical to 3D euclidian $(x,y,z) mapsto (r cos theta cos phi.alt , r cos theta sin phi.alt , r sin theta)$. Lenghts of the three sides of an infinitesimal cuboid whose diagonally opposite vertices are at $r, theta, phi.alt$ and $(r + d r, theta + d theta, phi.alt + d phi.alt)$ are $(d r, r d theta, r cos theta d phi.alt)$. Volume is $r^2 cos theta d r d theta d phi.alt$. \
  Determinant of jacobian $|(partial (x, y, z))/(partial (r, theta, phi.alt))| = r^2 cos theta$. \
  Density on spherical coordinates $p(r, theta, phi.alt)$ $->$ density on Euclidian coordinates is $p(x, y, z) = p(r, theta, phi.alt) |(partial (x, y, z))/(partial (r, theta, phi.alt))|^(-1)$. Infinitesimal probability mass of the cuboid above is equal to the mass of a Euclidian cuboid of size $(d x, d y , d z)$ at $(x, y, z)$.
]

== Normalizing Flows

Transform simple distribution $bold(z) tilde cal(N)(bold(0), I)$ through invertible map $T$ to get complex $bold(x) = T(bold(z))$.
Exact likelihood (no ELBO like VAEs), easy sampling.

*Change of Variables Formula:* \
$p_x (bold(x)) = p_z (T^(-1)(bold(x))) dot |det J_(T^(-1))(bold(x))|$, $|det J_(T^(-1))(bold(x))| = 1 / (|det J_T (T^-1(bold(x)))|)$.

*Diffeomorphism:* $T$ is bijective, differentiable, with differentiable inverse. Guarantees $det J_T eq.not 0$.

*Computational problem:* Computing $det J$ is $O(d^3)$ for dense Jacobian.
*Solution:* Design $T$ s.t. Jacobian is *triangular*, then only $O(d)$!

*Two architectures with triangular Jacobians:* 
#v(-6pt)
#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt,
  [], [*MAF*], [*IAF*],
  [Fast / parallel], [Density eval], [Sampling],
  [Slow / sequential], [Sampling], [Density ]
)

*Coupling layers*: Trick that makes both directions fast, at the cost of being less expressive per layer.

== Autoregressive Models

$p(bold(x)) = product_(i=1)^d p(x_i | bold(x)_(< i))$.

== Generative Adversarial Networks

Likelihood-free generative model: train via adversarial game between two networks:
*Generator* $G_theta$ maps latent $z tilde.op p_z$ (typically Gaussian) to fake samples; *Discriminator* $D_phi$: outputs prob that input is *real*.

*GAN Objective:*
$min_theta max_phi underbrace(EE_(x tilde.op p_r)[log D_phi (x)], "real samples") +$ \
#v(-16pt)
$underbrace(EE_(z tilde.op p_z)[log(1 - D_phi (G_theta (z)))], "fake samples")$
- *Discriminator* maximizes: correctly classify real (high $D$) and fake (low $D$)
- *Generator* minimizes: fool discriminator (make $D(G(z))$ high)

#colorbox(color:purple)[Common alternative objective for the generator is to maximize $EE_(z ~p(z))[log D(G(z))]]$ instead of minimizinh $EE_(z ~ p(z)) [log(1-D(G(z)))]$ to help mitigate vanishing gradient problem when discriminator becomes to good, i.e. $D(G(z)) -> 0$. \
For a fixed generator $G$, *optimal discriminator* $D^*$ is given by $D^* (x) = (p_"data" (x))/(p_"data" (x) + p_G (x))$. \
If *discriminator optimal*, GAN objevtive reduces to $V(D^*, G) = 2 D_"JS" (p_"data" || p_G)-log 4$.
]

=== Theoretical Foundation

Binary classification with $p(y=1) = p(y=0) = 1/2$:
- $y=1$: sample from real $p_r(x)$
- $y=0$: sample from generator $p_theta(x)$

*Bayes Optimal Classifier* (prob that $x$ is real): \
$q_theta (x) = P(y=1|x) = (p_r (x)) / (p_r (x) + p_theta (x))$.

*Generator Logistic Objective = JS Divergence:* \
$ell^* (theta) = EE_(tilde(p)_theta (x,y))[y ln q_theta (x) + (1-y) ln(1 - q_theta (x))]$\ $= "JS"(p_r || p_theta) - ln 2$.

*Jensen-Shannon Divergence:*
$"JS"(p_r || p_theta) = 1/2 D_"KL" (p_r || p_m) + 1/2 D_"KL" (p_theta || p_m), quad p_m = (p_r + p_theta)/2$. \
*Bounded:* $0 <= "JS"(p_r || p_theta) <= log 2$.

#colorbox(color:purple, title: "Jensen Inequality", inline:false)[
  If $phi$ convex: $phi(EE[X]) <= EE[phi(X)]$. If concave, other way around.
]

=== Training
Alternating SGD (heuristic, may diverge!).
Training is *Saddle-point problem*, notoriously unstable!

*JS Divergence Saturates* when distributions don't overlap.
If $p_r$ and $p_theta$ have disjoint supports: discriminator perfect, no gradient for generator!

*Wasserstein Distance (Earth Mover's Distance):* \
$W(p_r, p_theta) = inf_(gamma in Pi(p_r, p_theta)) EE_((x,y) tilde.op gamma) [||x - y||]$
Minimum total "work" to transport mass from $p_r$ to $p_theta$. Provides meaningful gradients even without overlap.

*Dual (Kantorovich-Rubinstein):* \
$W(p_r, p_theta) = sup_(||f||_L <= 1) EE_(x tilde.op p_r)[f(x)] - EE_(x tilde.op p_theta)[f(x)]$. \
Maximize gap between avg score of real vs fake samples w.r.t. Lipschitz constraint.\ Max achievable gap $=$ Wasserstein distance.

*WGAN* uses critic $f_w$ (not classical discriminator!): \
$min_theta max_w EE_(x tilde.op p_r)[f_w (x)] - EE_(z tilde.op p_z)[f_w (G_theta (z))]$. \
*Enforcing Lipschitz:*
- *Weight clipping* (original): crude, problematic
- *Gradient penalty*: add $lambda EE_(hat(x))[(||nabla_(hat(x)) f_w (hat(x))||_2 - 1)^2]$ 
*Mode Collapse:* Generator produces only few samples that fool discriminator, ignoring full distribution diversity.

== Diffusion Models

*Forward process (fixed):* Gradually add Gaussian noise over $T$ steps until data becomes pure noise.

Fwd step: $q(x_t | x_(t-1)) = cal(N)(x_t; sqrt(1 - beta_t) x_(t-1), beta_t I)$.
Full fwd proces: $q(x_(1:T) | x_0) = product_(t=1)^T q(x_t | x_(t-1))$.

*Noise schedule:* ${beta_t in (0,1)}_(t=1)^T$ noise added at each step.
*Define*: $alpha_t = 1 - beta_t$ and $overline(alpha)_t = product_(i=1)^t alpha_i$.

*Direct sampling (reparameterization trick):*
#align(center)[
$q(x_t | x_0) = cal(N)(x_t; sqrt(overline(alpha)_t) x_0, (1 - overline(alpha)_t) I)$. \
$x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1 - overline(alpha)_t) epsilon_0, quad epsilon_0 tilde.op cal(N)(0, I)$.
]

*Reverse process (learned):* Train NN to denoise step by step: $p_theta (x_(t-1)|x_t) = cal(N)(x_(t-1); mu_theta (x_t, t), sigma_t^2 I)$.


For small $beta_t$, the reverse $q(x_(t-1)|x_t)$ is also Gaussian.

$log p_theta (x_0)$ intractable, so derive *Variational Lower Bound (VLB)*:
$-log p_theta (x_0) <=
cal(L)_"VLB" = EE_q [log (q(x_(1:T) | x_0)) / (p_theta (x_(0:T)))]$.
*Decomposition into 3 terms:*
#text(size:0.9em)[
$cal(L)_"VLB" = underbrace(D_"KL" (q(x_T|x_0) || p(x_T)), L_T) + $\ $sum_(t=2)^T underbrace(EE_(q(x_t|x_0)) [D_"KL" (q(x_(t-1) | x_t, x_0) || p_theta (x_(t-1) | x_t))], L_(t-1))$\
$-underbrace(EE_(q(x_1|x_0)) [log p_theta (x_0 | x_1)], L_0)$]
- $L_T$: Is $q(x_T | x_0) approx cal(N)(0, I)$? Not optimized.
- $L_(t-1)$: Match learned reverse to true reverse
- $L_0$: Reconstruction term

*Tractable Reverse Posterior* $q(x_(t-1) | x_t, x_0)$ is Gaussian with closed form (product of Gaussians):
$q(x_(t-1) | x_t, x_0) = cal(N)(x_(t-1); mu_(q,t)(x_t, x_0), sigma_t^2 I)$, with:

$mu_(q,t)(x_t, x_0) = 1/sqrt(alpha_t) (x_t - (1 - alpha_t)/sqrt(1 - overline(alpha)_t) epsilon_0)$,
$sigma_t^2 = (1 - overline(alpha)_(t-1))/(1 - overline(alpha)_t) beta_t$.

== Noise Prediction Parameterization

Predict the *noise* $epsilon_theta (x_t, t)$ instead of mean directly.
Parameterize learned mean to mirror true posterior:
$mu_theta (x_t, t) = 1/sqrt(alpha_t) (x_t - (1 - alpha_t)/sqrt(1 - overline(alpha)_t) epsilon_theta (x_t, t))$.
Since both distributions are Gaussian with same variance:
$D_"KL" (cal(N)(mu_q, sigma^2 I) || cal(N)(mu_p, sigma^2 I)) = 1/(2 sigma^2) ||mu_q - mu_p||^2$.
This simplifies $L_(t-1)$ to comparing noise:
$L_(t-1) = EE_(x_0, epsilon_0) [(1-alpha_t)^2 / (2 alpha_t (1-overline(alpha)_t) sigma_t^2) ||epsilon_0 - epsilon_theta (x_t, t)||^2]$. \
$cal(L)_"simple" = EE_(t tilde.op [1,T], x_0, epsilon_0) [||epsilon_0 - epsilon_theta (x_t, t)||^2].$

#table(
  columns: 2,
  stroke: 0.5pt,
  inset: 4pt,
  [Training], [Sampling], 
  [#text(0.65em)[
1. Sample real image $x_0 tilde.op q(x_0)$
2. Sample random timestep $t tilde.op "Uniform"({1, ..., T})$
3. Sample noise $epsilon tilde.op cal(N)(0, I)$
4. Compute noisy image: $x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1 - overline(alpha)_t) epsilon$
5. Grad step on $nabla_theta ||epsilon - epsilon_theta (x_t, t)||^2$
  ]],
  [#text(0.75em)[
1. Sample $x_T tilde.op cal(N)(0, I)$
2. For $t = T, ..., 1$: \
#v(-5pt)
#text(size:0.7em)[
$z tilde.op cal(N)(0, I)$ if $t > 1$, else $z = 0$ \
$x_(t-1) = 1/sqrt(alpha_t) (x_t - (1-alpha_t)/sqrt(1-overline(alpha)_t) epsilon_theta (x_t, t)) + sigma_t z$] \
Return $x_0$
  ]],
)

Cosine noise schedule performs better than linear.

Used architecture is U-Net.
*Input:* Noisy image $x_t$ + timestep $t$;
*Output:* Predicted noise $epsilon_theta (x_t, t)$.

Model *conditional distribution* $p_theta (x_(0:T) | y)$ where $y$ is condition (class, text, image).
Extend denoiser to take $y$ as input.

*Latent Diffusion Models (LDM)*
run diffusion in *compressed latent space* instead of pixel space.

= Tricks

*Short Connections* in DNs: Add less deep paths to a very deep network. *Residual* connections: shortcut and add back in. *Skip* connections: concatenate.

#colorbox(color:purple)[
  For $y_1 = alpha f(x, Theta_1) + x$, $x in RR^d, y_i in RR^d, Theta_i in RR^(d times d)$ and $y_(i>=2)=alpha f(y_(i-1), Theta_i) + y_(i-1)$, $l=L(y_n)$
  it holds that $(partial y_k)/(partial y_(k-1)) = alpha (partial f(y_(k-1), Theta_k))/(partial y_(k-1)) + I_d$ and $(partial y_k)/(partial Theta_k) = alpha (partial f(y_(k-1), Theta_k))/(partial Theta_k)$. By applying the chain rule we have $(partial l)/(partial Theta_k) = (partial L(y_n))/(partial y_n) (partial y_n)/(partial y_(n-1)) ... (partial y_(k+1))/(partial y_k) (alpha partial f(y_(k-1), Theta_k))/(partial Theta_k)$. Set $alpha^2 = a/n$ for $a>0$ s.t. $lim_(n -> infinity)EE norm(y)^2 < infinity$.
]

== Weight Decay & Early Stopping

*L2 regularization* \
$cal(R)_Omega (theta; cal(S)) = cal(R)(theta; cal(S)) + Omega (theta)$, $Omega_mu (theta) = mu/2 norm(theta)^2$, $mu >= 0$. \
Only penalize weights, not biases.
*GD upd w/ WD:*
$Delta theta = -eta nabla cal(R)(theta) - eta nabla Omega_mu (theta) = -eta nabla cal(R)(theta) - eta mu theta$.

Geometric interpration (local quadratic approx):
Regularized optimum: $theta_mu^* = (H + mu I)^(-1) H theta^"*"$, where $H = Q^top Lambda Q$ gives $theta_mu^* = Q "diag"(lambda_i / (lambda_i + mu)) Q^top theta^*$.

$lambda_i >> mu$:$lambda_i/(lambda_i + mu) approx 1$$→$weak shrinkage (important dirs). \
$lambda_i << mu$:$lambda_i/(lambda_i + mu) approx 0$$→$strong shrinkage (flat dirs).

Adaptively shrinks based on loss geometry, preserves important dirs, removes unnecessary complexity.

*Early stopping*: Rather than training to convergence, stop when validation performance plateaus. Analysis shows that this is approximately equivalent to L2 regularization. GD trajectories can be approximated as
$theta(k) = [I- (I-eta Lambda)^k] theta^*$. For small step sizes, behaves like weight decay when $k=1/(eta mu)$.

#colorbox(color: purple)[
  $L^1$ regularized second-order approximation of an arbitrary loss function around optimal $theta^*$ is $R_(L^1)(theta) approx R(theta^*) + 1/2 (theta - theta^*)^top H (theta - theta^*) + lambda norm(theta)_1$.
  Assuming $H = "diag"(h_1, ... , h_d)$, we get $R_(L^1)(theta) approx sum_(i=1)^d [h_i/2 (theta_i - theta_i^*)^2 + lambda |theta_i| ] + "const"$ so we need to minize $f(a) = 1/2 (a-b)^2 + beta |a|$ and this gives $a^* = "sgn"(b) max {0, |b| - beta}$, so $theta_i = "sgn"(theta_i^*) max {0, |theta_i^*| - lambda / h_i}$. For $L^2$ regularization we get $theta_i = h_i/(h_i + lambda) theta_i^*$. *Connecting $L^2$ w early stopping*: 
  $R(w) approx R(w^*) + 1/2 (w-w^*)^top H (w-w^*)$. $nabla R(w) = H (w-w^*)$.
  GD update: $w^t = w^(t-1) - eta H(w^(t-1)-w^*)$ gives
  $w^t - w^*= (I_d - eta H) (w^(t-1)-w^*)$. Using $H=Q Lambda Q^top$: $Q^top (w^t - w^*) = (I_d - eta Lambda) Q^top (w^(t-1) - w^*)$. If $w^0 = 0$, $Q^top (w^t - w^*) = (I_d - eta Lambda)^t Q^top (0 - w^*) => Q^top w^t = [I_d - (I_d - eta Lambda)^t]Q^top w^*$. Optimal $w$ under $L^2$ reg gives 
  $Q^top w = [I_d - lambda (Lambda + lambda I_d)^(-1)] Q^top w^*$. Matching both gives $t approx 1/(eta lambda)$.
  *Weight normalization* is like BatchNorm,with the covariance matrix replaced by the identity matrix.
]

== Ensemble Methods

*Bagging*: Create $K$ bootstrap samples of Data (sampling with replacement), train separate models, and average predictions: $p(y|x) = 1/K sum_(k=1)^K p(y | x; theta_k)$.

*Dropout*: Randomly drop units during training with probability $1-pi$ . Creates an exponential ensemble of sub-networks sharing weights. Test time: Scale weights by $pi$ to approximate the ensemble average.

== Normalization

*Batch Norm*: Normalize activations across mini-batch:
$tilde(z) = (z-mu_"batch") / sigma_"batch"$, $hat(z) = alpha tilde(z) + beta$, $mu_"batch" = 1/b sum_(i=1)^b z_i$, $sigma_"batch" = sqrt(1/b sum_(i=1)^b (z_i - mu_"batch")^2)$. $quad$ *Layer Norm*: Normalize features in a layer instead; particularly effective for RNNs (batch statistics are less stable).

== Data / Task Augmentation

Augment Data by applying valid transformations. \
Semi-supervised Learning: Train jointly on labeled and unlabeled data w combined loss. Pre-training & Fine-tuning.
Multi-task Learning. Self-supervised Learning: Create free supervision from data.

= Recurrent Neural Networks

*Evolution:* $z_t = F[theta](z_(t-1), x_t)$, with $z_0 = 0$. \
Optional output: $y_t = G[theta](z_t)$.

*Simple RNN:* $z_t = phi.alt(W z_(t-1) + U x_t)$ where $W in RR^(m times m)$, $U in RR^(m times n)$

*Backpropagation Through Time* (param sharing):
$(diff R)/(diff w_(i j)) = sum_t (diff R)/(diff z_(i)^t) dot.c dot(phi.alt)_(i)^t dot.c z_(j)^(t-1)$. \
$(diff R)/(diff u_(i k)) = sum_t (diff R)/(diff z_(i)^t) dot.c dot(phi.alt)_(i)^t dot.c x_(k)^t$;
$dot(phi.alt)^t_i = phi.alt'(F_i (z^(t-1), x^t))$.

*Gradient flow backward through time:* \
$nabla_(x_t) cal(R) = [product_(r=t+1)^s W^top S(z^r)] dot.c J_G dot.c nabla_y cal(R)$

#text(0.99em)[
*Spectral analysis:* $norm(product W^top S(z^r))_2 <= [sigma_(max)(W)]^(s-t)$
]
*Root cause:* Repeated matmul through time. \
$=>$ Simple RNNs cannot learn long dependencies.

*Deep RNNs* stack layers vertically: \
$z^(t,ell) = phi(W_ell z^(t-1,ell) + U_ell z^(t,ell-1))$
where $z^(t,0) = x_t$.

#colorbox(color: purple)[
  For RNN with $z_(t+1) = phi (U z_t + V x_(t+1))$, $L= sum_(t=1)^T ell (hat(y)_t, y_t)$, where $hat(y)_t$ depends on $z_t$. Then
  $(partial L)/(partial U) = sum_(t=1)^T (partial L)/(partial z_t) dot (phi_t^' dot z_t)$, \
  $(partial L)/(partial V) = sum_(t=1)^T (partial L)/(partial z_t) dot (phi_t^' dot x_(t+1))$. \
  *Weight Sharing in RNNs (LSTM):*  \
  $(partial L)/(partial W) = sum_(t=1)^T (partial L)/(partial W_t)$. \
*Proof idea:* Introduce dummy parameters $tilde(W)_i = f(W)$ for each time step. By chain rule: $(diff L)/(diff W) = sum_i (diff L)/(diff tilde(W)_i) (diff tilde(W)_i)/(diff W)$. With constraint $tilde(W)_i = W$, we have $(diff tilde(W)_i)/(diff W) = I$, giving the sum. \
Initialization of bias in RNNs: Use 1.
]


== Long Short-Term Memory (LSTM)
- $C_t$: cell state (internal memory, protected highway)
- $z_t$: hidden state (external output, filtered view)
$C_t = underbrace(sigma(F tilde(x)^t) dot.o C_(t-1),"forget") + underbrace(sigma(G tilde(x)^t) dot.o tanh(V tilde(x)^t),"input")$, \
$z_t = underbrace(sigma(H tilde(x)^t) dot.o tanh(C_t),"output")$, where $tilde(x)^t = [x_t, z_(t-1)]$.



== Gated Recurrent Unit (GRU)

*Single state* $z_t$. *Input:* $tilde(x)^t = [x_t, z_(t-1)]$. \
$u_t = sigma(U tilde(x)^t), quad r_t = sigma(R tilde(x)^t)$, \
#text(0.99em)[
$z_t = u_t dot.o z_(t-1) + (1 - u_t) dot.o tanh(W [r_t dot.o z_(t-1), x_t])$
]

Often comparable to LSTM with fewer resources.
Gating creates identity paths $->$ better gradient flow.

== Linear Recurrent Models
RNNs not parallelizable during training. LRU has linear dynamics:
$z_(t+1) = A z_t + B x_t$. Diagonalize to
$A = P Lambda P^(-1)$, $lambda_i in CC$, change basis $zeta_t = P^(-1) z_t$. Then:
$zeta_(t+1) = Lambda zeta_t + C x_t$.
Each dimension evolves independently (no channel mixing). Compensate with expressive output: $y_t = "MLP"("Re"(G z_t))$.

*Stability:* Require $max|lambda_j| <= 1$ (spectral radius $<= 1$).

*Parameterization:* $lambda_i = exp(-exp(nu_i) + i phi_i)$ ensures $|lambda_i| in (0,1)$ automatically, $|lambda_i| approx 1$: Long-term memory, $|lambda_i| approx 0$: Short-term patterns. \
*Provably universal* as sequence-to-sequence map.

=== Connectionist Temporal Classification
*Problem:* Unsegmented sequences (e.g., speech). \
*Solution:* RNN outputs prob distribution over vocabulary at each time step.
Model all alignments with blank symbol "`-`":
$p(ell | x) = sum_(pi in cal(B)^(-1)(ell)) product_t y_(pi_t)$. \
$cal(B)$ removes blanks and repeated symbols.



== Sequence Learning

*Teacher Forcing:* $p(y^t)$ depends on $y^(1:t-1)$ only through $z^t$, means during autoregressive generation, model doesn't see its own predictions. 
*Solution:* Add feedback connections from $y^(t-1)$ to $z^t$: 
$z^t = "RNN"(z^(t-1), x^t, y^(t-1))$,
now model conditions on its own previous predictions → more coherent gen.

*Professor Forcing:* Train two networks (teacher-forced + free-running), discriminator matches hidden states → improved generalization.

*Exposure bias*: Model relies on itself where inputs come from the previous output because of the non-availability of the ground truth.

*Seq2Seq*: Input and output sequences have different lengths: Use encoder-decoder framework.


#colorbox(color: purple)[
  Gradients in bi-directional RNNs are computed by making a forward and backward run, then at timestep $t$ we combine (concatenate/add) and continue with the backpropagation. This happens at every bidirectional layer.
]

= Attention and Transformers

*Seq2Seq with Attention*: Encoder generates hidden state sequence. Decoding RNNs output attends to encoder states and gets used as input in next step.

*Attention*: Learn to index, multiplicative gating to combine bottom-up and top-down information.

KV - attention map: $F(bold(xi), ((bold(x)_1, bold(z)_1)), ..., (bold(x)_s, bold(z)_s)) = [bold(z)_1, ..., bold(z)_s] dot f(bold(xi), (bold(x)_1, ..., bold(x)_s))$ consisting of a query $bold(xi)$ (what to look for?), keys $bold(x)_i$ (index) and values $bold(z)_i$.

*Scaled Dot-Product Attention*: $f(bold(xi), bold(x)) = (bold(xi) dot bold(x))/sqrt(n)$

*Multi-headed attention*:
#v(-10pt)
#align(center)[
  $G(bold(xi), (bold(x)^t, bold(z)^t)_(t=1)^s) = bold(W) mat(
  F_1(xi, (bold(x)^t, bold(z)^t));
  dots.v;
  F_h(xi, (bold(x)^t, bold(z)^t))
),$
]
where #text(0.98em)[
$F_j (bold(xi), (bold(x)^t, bold(z)^t)) = F(bold(W)_j^q xi, (bold(W)_j^x bold(x)^t, bold(W)_j^z bold(z)^t))$
].

$text("Attention")(Q, K, V)
= "softmax"((Q K^T) / sqrt(d_k)) V$, where \
$Q = X W_Q, quad K = X W_K, quad V = X W_V$ and
$X in RR^(T times d_"model")$,
$W_Q, W_K in RR^(d_"model" times d_k)$,
$W_V in RR^(d_"model" times d_v)$,
$Q, K in RR^(T times d_k)$,
$V in RR^(T times d_v)$.


*Positional encodings* necessary since no use of recurrence. Can use predefined appraoches, or learned.

Self attention used in encoder, masked self-attention in decoder. Can also add cross-attention to decoder.

*ELMo* consists of multiple layers of 2 LSTMs working in opposite directions. Can then be used to collapse all layers $2L + 1$ and train in task-specific manner.

*BERT* is trained on two simultaneous tasks (Masked Language Modeling and Binary Prediction whether sentence B follows A).
Has bidirectional encoder, and task-specific heads. Can do FT for various tasks.

*Vision Transformers* split images into patches, add pos embeddings and a [CLS] token, then process with a standard transformer encoder.

#colorbox(color:purple)[
#text(size:0.8em)[
*Complexity metrics for different layer types in terms of input sequence length $n$.*

*Key Metrics:*
- *Complexity per Layer*: Total computational operations per layer
- *Sequential Operations*: Number of operations needed to connect any two input positions
- *Maximum Path Length*: Longest path between any two input positions in the network

*Self-Attention:*
- Complexity per Layer: $cal(O)(n^2 d)$ — quadratic in sequence length due to all-pairs attention
- Sequential Operations: $cal(O)(1)$ — fully parallelizable
- Maximum Path Length: $cal(O)(1)$ — direct connections between all positions

*RNN:*
- Complexity per Layer: $cal(O)(n d^2)$ — linear in sequence length
- Sequential Operations: $cal(O)(n)$ — must process sequentially
- Maximum Path Length: $cal(O)(n)$ — information flows through entire sequence

*Trade-off:* Self-attention enables parallel processing and direct long-range connections but has quadratic complexity, while RNNs have linear complexity but require sequential processing.
]]


#text(size:0.7em)[
= Ethics

*Adversarial examples* (given $f(x) = y$ correctly):\
- Untargeted: $norm(delta)<=epsilon$ s.t. $f(x + delta) != y$.
  - Optimize $max_(norm(delta) <= epsilon) L(f(x+delta), y)$.
- Targeted: $norm(delta)<=epsilon$ s.t. $f(x + delta) = t != y$.
  - Optimize $min_(norm(delta) <= epsilon) L(f(x+delta), t)$.

*Linear Binary* ($y in {-1, 1}, f(x) = w^top x + b$):\
*Correct*: $y (w^top x + b) > 0$. \
Adv. flips when $y w^top delta <= -y(w^top x + b)$ cross hyperplane.
*L2 optimal*: $delta^* = (-w^top x + b) / norm(w)_2^2 w$, $norm(delta^*)_2 = (|w^top x + b|) / norm(w)_2$. \
$L_infinity$ optimal: $delta = -epsilon "sign"(y w)$. \
*Multiclass*: $f_k (x) = w_k^top x + b_k$, use $"argmax"_k f_k (x)$. \
#text(0.96em)[
*Margin* to class $j$: $m_j (x) = (w_y - w_j)^top x + (b_y - b_j)$.] \
$f_y (x) = f_j (x) <=> m_j (x) = 0$. 
*Correct* if $f_y (x) > f_j (x) space forall j != y$, *adversarial* if $exists j != y$ s.t. $f_y (x + delta) < f_j (x + delta)$.
Distance to boundary: $(m_j (x)) / norm(w_y - w_j)_2$. \
Adversarial attacks for *NNs*: Approximate boundary by $f(x + delta) approx f(x) + nabla f(x)^top delta$. *FGSM* is a one-step $L_infinity$ attack: $delta = epsilon "sign"(nabla_x L(f(x), y))$. *PGD* is multi-step $delta_{t+1} = "Proj"_(norm(delta) <= epsilon) (delta_t + alpha "sign"(g_t))$. \
*Distributionally Robust Optimization*: \
$"min"_f sup_(Q in U(P)) E_Q [L(f(x))]$, where $U$ means close.
Can use upper bound on Wasserstein distance e.g. \
*Robust training* $min_f EE[max_(delta in S) L(f(x + delta), y)]$. \
Adversarial training can be viewed as robustness to distribution shift measured by Wasserstein distance. \
*Interpretability*: Local - explain pred for specific $x$, Global - explain model behaviour on avg over data. \
*Local*: Ceteris paribus (vary $x_j$, fix $x_(-j)$), Sensitivity ($diff_(x_j) f(x) $), missing info ($f(x) - EE[f(X) | X_(-j) = x_(-j)]$). *Global*: Mutual info ($I(X_j ; Y [ | X_(-j)])$), Predictive util (train $f$ w/ and w/o $x_j$). For log-loss predictive util $approx$ conditional mutual information. \
SHAP attributes predictions, while SAGE attributes risk reduction. \
$A$ protected attribute, $Y$ target outcome, $hat(Y)$ prediction. Demographic Partiy: $hat(Y) bot A$; Equalized Odds: $hat(Y) bot A | Y$, Equality of Opportunity: $hat(Y) bot A | Y=1$.]