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
$I = union.big_(i = 1)^N R_i$ with $R_i sect R_j = emptyset quad forall i != j$.

#colorbox(title: [Thresholding], color: silver, inline: true)[
  Segment image into 2 classes. \
  $B(x, y) = 1 "if" I(x, y) >= T "else" 0$, finding $T$ with trial and error, compare results with ground truth.
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