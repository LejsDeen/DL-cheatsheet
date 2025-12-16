#import "template.typ": *
#import "@preview/diagraph:0.2.1": *
#import "@preview/xarrow:0.4.0": xarrow

// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.
#show: project.with(
  title: "Deep Learning Cheat Sheet",
  authors: (
    (name: "Authors"),
  ),
  date: "January 2026",
)

#v(1000pt)
#h(1pt)
#v(1000pt)
#h(1pt)
#v(1000pt)

= Computer Vision

== The Digital Image & Sensors

#colorbox(title: [Charge Coupled Device (CCD)], inline:false, color:gray)[
Photons accumulate in individual sensor elements (buckets) during exposure time (charge proportional to light intensity).
  - *Blooming*: Oversaturated photosites cause vertical channels to "flood" (bright vertical line)
  - *Bleeding/Smearing*: While charge in transit, bright light hits photosites above
  - *Dark current*: CCDs produce thermally generated charge: random noise despite darkness. Avoid by cooling or subtracting a dark frame.
]
*Bilinear interpolation*: Easy way to reconstruct 2D signal, assuming $f$ behaves linearly between sampled points.
$accent(f,hat) (x,y) = (1-a)(1-b)F_(i,j)+a(1-b)F_(i+1,j)+a b F_(i+1,j+1)+(1-a)b F_(i,j+1)$.

#colorbox(title: [Image Noise], inline:false)[
  Additive Gaussian noise:
  $I(x,y)=f(x,y)+c$, \ where $c tilde cal(N)(0,sigma ^2)$ and 
  $p(c)= (2 pi sigma ^2)^(-1)exp((-c^2)/(2 sigma ^2))$. \
  Poisson / shot noise: $p(k) = (lambda^k e^(- lambda)) / (k!)$. \
  Rician / MRI noise: $p(I) = I / (sigma^2) exp((-(I^2 + f^2)) / (2 sigma^2)) I_0 ((I f) / (sigma^2))$. \
  Multiplicative noise: $I = f + f c$. \
  Signal to noise ratio (SNR), $s$ is index of image quality: $s = F / sigma$,  $F = 1 / (X Y) sum_(x = 1)^X sum_(y = 1)^Y f(x, y)$. PSNR: $s_"peak" = (F_"max") / sigma$.
]

*Color camera concepts:*
1. Prism (split light, 3 sensors, needs good alignment, good color separation)
2. Filter mosaic (Coat filter on sensor, Demosaicing to obtain full color, introduces aliasing)

== Image Segmentation
Pixel-wise classification problem, to group pixels in an image that share common properties. \
Segmentation of $I$: Find $R_1, ..., R_n$ such that \
$I = union.big_(i = 1)^N R_i$ with $R_i sect R_j = emptyset quad forall i != j$.

#colorbox(title: [Thresholding], color: silver, inline: true)[
  Segment image into 2 classes. \
  $B(x, y) = 1 "if" I(x, y) >= T "else" 0$, finding $T$ with trial and error, compare results with ground truth.
]
#colorbox(title: "Convolution")[
  Point spread function: $I' = K convolve I$. \
  $I'(x, y) = sum_((i, j) in N(x, y)) K(i, j) I(x - i, y - j)$. \
  Linear, associative, shift-invariant, commutative (if dims are identical). *$(f * g)(t) = integral_RR f(a) g(t - a) dif a$*
]

#colorbox(title: "Important Kernels", color: purple, inline: true)[
  
  #set math.mat(delim: "[")
  #v(-7pt)
  #grid(columns: (auto, auto, auto, auto), gutter: 1em,
    [Laplacian], [$"Prewitt"_x$], [#v(-10pt) Low-pass/ \ Mean / Box], [High-pass], 
    $mat(0,1,0; 1,-4,1;0,1,0)$,
    $mat(-1,0,1;-1,0,1;-1,0,1)$,
    $1 / 9 mat(1,1,1; 1,1,1; 1,1,1)$,
    $mat(-1,-1,-1;-1,8,-1;-1,-1,-1)$,
    [Gaussian], [$"Sobel"_x$], [$"Diff"_x$], [$"Diff"_y$],
    [\ $ 1 / (2 pi sigma^2) e^(-(x^2 + y^2) / (2 sigma^2))$],
    $mat(-1,0,1; -2,0,2; -1,0,1)$,
    $mat([-1], 1)$,
    $mat([-1], 1)^top$
  )

]

*Dirac delta*: $delta(x) = cases(0 "if" x!=0, "undefined else")$ with $integral_(-oo)^infinity delta(x) dif x = 1$. $cal(F)[delta(x - x_0)](u) = e^(-i 2 pi u x_0)$. $delta(u) = integral_RR e^(-i 2 pi x u) dif x$.\
*Sampling* $f$ at points $x_n$: $f_("s")(x)=sum_(n)f(x_n)delta(x-x_n)$.
#v(-3pt)
#grid(columns: (auto, auto, auto), column-gutter: 1.5em, row-gutter: 0.3em,
  [*Property*], $bold(f(x))$, $bold(F(u))$,
  [Linearity], $alpha f_1(x) + beta f_2(x)$, $alpha F_1(u) + beta F_2(u)$,
  [Duality], $F(x)$, $f(-u)$,
  [Convolut.], $(f * g)(x)$, $F(u) dot.c G(u)$,
  [Product], $f(x) g(x)$, $(F * G)(u)$,
  [Timeshift], $f(x - x_0)$, $e^(-2 pi i u x_0) dot.c F(u)$,
  [Freq. shift], $e^(2 pi i u_0 x) f(x)$, $F(u - u_0)$,
  [Mult], $f(a t)$, $1/norm(a) F(u/a)$,
  [MatMul], $f(A t)$, $1/norm(det(A)) F(A^(- top)u)$ 
)

#grid(columns: (60%, 39%), column-gutter: 0.4em, image("fourier-transforms.png", height: 16.6em), [*Simple procedure of sampling and reconstructing a 2D signal*: Sample Signal, $"FT"$, Cut out Magnitude Spectrum by multiplication with box filter, $"FT"^(-1)$. 
*Some reconstruction filters*: Nearest neighbor, Bilinear (equiv. to convolving sampled signal twice w/ box filter),])
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