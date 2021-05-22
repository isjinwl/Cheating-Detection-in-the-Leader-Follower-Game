## Convolutional Neural Network(CNN)
----
>**Neural Network and Deep Learning(邱锡鹏，2020)**[[Book]](https://nndl.github.io/)<br/>
> #### Appendix.A 线性代数(Linear Algebra)
> + N维向量
>    + 标量(Scalar)：是一个实数，只有大小，没有方向。e.g.,*a,b,c,...*
>    + 向量(Vector)：是一组实数构成的有序数组，有大小，有方向.e.g.,N维向量 ***a***=[$a_1$, $a_2$, ..., $a_N$]
>    + 向量空间(Vector Space or Linear Space)：向量组成的集合，且满足*向量加法*和*标量乘法*（两者可视为向量空间的条件）.
>       + 欧氏空间(Euclidean Space)：一个欧氏空间通常表示为$\mathbb{R}^N$，N为空间维度(Dimension).
>          + 欧氏空间中的*向量加法*: <br/>
***a***+***b***=[$a_1, a_2, ..., a_N$]+[$b_1, b_2, ..., b_N$]=[$a_1+b_1,a_2+b_2,...,a_N+b_N$]
>          + 欧氏空间中的*标量乘法*：<br/>
*c* $\cdot$ ***a*** = *c* $\cdot$ [$a_1, a_2, ..., a_N$]=[*c*$a_1$, *c*$a_2$, ..., *c*$a_N$]. 其中*a,b,c* $\in \mathbb{R}$ 属于标量.
>       + 线性子空间：向量空间的一个子集，也满足*向量加法*和*标量乘法*.
>       + 线性相关(linearly dependent)：如果向量空间中的一个向量可以用有限个其他向量的线性组合表示，则称线性相关.<br/>或线性空间$\mathbb{v}$中的**M**个向量{$v_1, v_2,...,v_M$}，如果$\lambda_1 v_1+\lambda_2 v_2+...+\lambda_M v_M=0$存在非零解$\lambda_1,\lambda_2,...,\lambda_M$（标量），则向量组{$v_1, v_2,...,v_M$}是线性相关的；否则向量组{$v_1, v_2,...,v_M$}是线性无关的(linearly independent).
>        + 基向量：N维向量空间$\mathcal{V}$的基(Base)$\mathcal{B}$={$e_1,e_2,...,e_N$}是$\mathbb{v}$的有限子集，其元素之间线性无关。向量空间$\mathbb{v}$的所有向量都可以按唯一的方式表达为$\mathcal{B}$中向量的线性组合，即对于任意 $v \in \mathcal{V}$存在一组标量{$\lambda_1,\lambda_2,...,\lambda_N$}，使得$v=\lambda_1e_1+\lambda_2e_2+...+\lambda_Ne_N$. 且如果基向量是有序的，则($\lambda_1,\lambda_2,...,\lambda_N$)称为向量$v$关于基$\mathcal{B}$的坐标.如果基向量是Standard Basis,($\lambda_1,\lambda_2,...,\lambda_N$)则可称为向量$v$的笛卡尔坐标(Cartesian Coordinate).
>        + 内积(Inner Product|Dot Product|Scalar Product)：一个N维向量空间中的2个向量***a***和***b***的内积为$<a,b>=\sum_{n-1}^Na_nb_n$(基向量相同,cos$\theta$=1).
>        + 正交(Orthogonal)：如果同一个向量空间中2个向量的内积为0，则这两个向量正交；如果向量空间A中的一个向量$v$与子空间$\mathcal{U}$的每个向量正交，那么向量$v$与子空间$\mathcal{U}$正交.
>    + 范数(Norm)：一个表示向量“长度”的函数，为向量空间内的所有向量赋予*非零*的正长度或大小.对于一个N维向量$\mathcal{v}$, 常见的一个范数函数$\mathcal{l_p}$范数，$\mathcal{l_p}(\mathcal{v}) = \begin{Vmatrix} \mathcal{v} \end{Vmatrix}_p=(\sqrt{\sum_{n=1}^N\begin{Vmatrix}v_n\end{Vmatrix}^p})^{\frac 1p}$. 其中$p\ge 0$为这个标量的参数，常见取值有$1,2,\infty$.
>        + $\mathcal{l_1}$范数：$\begin{Vmatrix} \mathcal{v} \end{Vmatrix}_1=\sum_{n=1}^N \begin{vmatrix} \mathcal{v_n} \end{vmatrix}.$ 向量的各个元素的绝对值之和.
>        + $\mathcal{l_2}$范数：$\begin{Vmatrix} \mathcal{v} \end{Vmatrix}_2=\sqrt{\sum_{n=1}^N \mathcal{v_n^2}}=\sqrt{\mathcal{v^T}\mathcal{v}}.$ 向量的各个元素的平方和再开平方. $\mathcal{l_2}$范数又称Euclidean范数或者Frobenius范数或者向量的*模*.从几何角度，向量是从原点出发的一个带箭头的有向线段，$\mathcal{l_2}$范数是线段的长度。
>        + $\mathcal{l_\infty}$范数：$\begin{Vmatrix}\mathcal{v}\end{Vmatrix}_\infty$=max{$v_1,v_2,...,v_N$}. 向量的各个元素的*最大绝对值*.
>    + 常见向量：
>       + 全0向量：所有元素都为0的向量，表示为0，是笛卡尔坐标系中的原点.
>       + 全1向量：所有元素都为1的向量，表示为1.
>       + one-hot向量：有且只有一个元素为1，其他元素都为0的向量.
> + 矩阵
>    + 线性映射(Linear Mapping)：即线性变换，可理解为一个映射函数，把*向量*从线性空间$\mathcal{X}$映射到线性空间$\mathcal{Y}$，表示为$f:\mathcal{X} \to \mathcal{Y}$，并满足：<br/>
> 对于$\mathcal{X}$中任意两个向量$u$和$v$，以及任意标量$c$，有$f(u+v)=f(u)+f(v), f(c \cdot v)=cf(v)$.
>       + 例：函数$f:\mathbb{R}^N \to \mathbb{R}^M$<br/>$y=Ax={\begin{vmatrix}a_{11} & a_{12} & \cdots & a_{1N}\\a_{21} & a_{22} & \cdots & a_{2N}\\\vdots & \vdots & \ddots & \vdots\\a_{M1} & a_{M2} &  \cdots & a_{MN}\end{vmatrix}}\times{\begin{vmatrix}x_{1}\\x_{2}\\ \vdots\\x_N\end{vmatrix}}={\begin{vmatrix}a_{11}x_1+a_{12}x_2+\cdots+a_{1N}x_N\\a_{21}x_1+a_{22}x_2+\cdots+a_{2N}x_N\\ \vdots \\a_{M1}x_1+a_{M2}x_2+\cdots+a_{MN}x_N\end{vmatrix}}={\begin{vmatrix}y_1\\y_2\\ \vdots \\y_M\end{vmatrix}}.$<br/>$(A \in \mathbb{R}^{M \times N}, x \in \mathbb{R}^N, y \in \mathbb{R}^M).$  
>    + 仿射变换
> ----
> #### Appendix.B 微积分(Calculus)
> + 微分(Differentiation)
>    + 导数(Derivative)：定义一个函数$f:\mathbb{R} \to \mathbb{R}$ (定义域和值域都是实数域)，若$f(x)$在点$x_0$的某个邻域$\Delta x$内，极限$f'(x_0)=lim_{\Delta x \to 0}{{f(x_o+\Delta x)-f(x_0)}\over{\Delta x}}$存在，则称函数$f(x)$在点$x_0$可导，$f'(x_0)=\frac{df(x_0)}{dx}$成为其导数.
>       + 常见函数的导数：
>         |函数    |函数形式               |导数            |
>         |:------|:----------------------|:---------------|
>         |常函数  |$f(x)=C,C$为常数.      |$f'(x)=0$       |
>         |幂函数  |$f(x)=x^r,r$为非零实数.|$f'(x)=rx^{r-1}$|
>         |指数函数|$f(x)=exp(x)$          |$f'(x)=exp(x)$  |
>         |对数函数|$f(x)=log(x)$          |$f'(x)=\frac 1x$|  
>       + 高阶导数：对一个函数的导数继续求导. 例，二阶导数$f''(x)=f^{(2)}(x)={{d^{(2)}f(x)}\over{dx^2}}$.
>       + 偏导数(Partial Derivative)：多元变量函数$f:\mathbb{R}^D \to \mathbb{R}$关于其中一个变量$x_i$的导数，保持其他变量固定.例，一阶偏导$f_{x_i}'(x)=\nabla_{x_i}f(x)={\partial f(x) \over \partial x_i}={\partial \over \partial x_i}f(x)$.
>    + 微分及可微函数(Differentiation & Differentiable Function)：给定一个连续函数，计算其导数的过程称为*微分*. 如果一个函数$f(x)$在定义域内的所有点都存在导数，则$f(x)$称为*可微函数*. 可微函数$\Rightarrow$连续函数，连续函数则不一定可微. 例，函数$f(x)=\begin{vmatrix}x\end{vmatrix}$为连续函数，但在点$x=0$处不可导.
>    + 泰勒公式(Taylor's Formula)：
>       + 如果函数$f(x)$在点$a$处$n$次可导($n \ge 1$)，在一个包含点$a$的区间上的任意$x$都有$f(x)=f(a)+\frac{1}{1!}f'(a)(x-a)+\frac{1}{2!}f^{(2)}(a)(x-a)^2+..._\frac{1}{n!}f^{(n)}(a)(x-a)^n+R_n(x), n\ge 1$.
>       + 泰勒公式也是用来求解函数在特定点的邻域中的近似值，一阶泰勒展开可以视为变化，二阶视为变化的变化，依次求和所有变化，原文具体表述是‘已知函数$f(x)$在点$a$的各阶导数值，用这些导数值做系数构建一个多项式来近似该函数在点$a$的邻域中的值. 多项式部分称为函数$f(x)$在点$a$的$n$阶泰勒展开式，$R_n(x)$称为泰勒公式的余项，是$(x-a)^n$的高阶无穷小’. 
> + 积分(Integration)：积分是微分的逆过程. 微分是对连续函数求导数的过程，积分是从导数推算出原函数的过程.
>    + 不定积分(Indefinite Integral)：
>         + $F(x)=\int f(x)dx,f(x)$是$F(x)$的导数，$F(x)$是$f(x)$的原函数、不定积分，$dx$表示积分变量是$x$. 
>         + 函数$f(x)$的不定积分是不唯一的，根据导数性质，$F(x)+C$也是$f(x)$的不定积分($C$为常数).
>    + 定积分(Definite Integral)：
>       + $F(x)=\int_b^a f(x)dx,x \in [a,b]. F(x)$是$f(x)$的定积分.
>       + 可以理解为面积(在坐标平面上由函数$f(x)$，垂直直线$x=a,x=b, x$轴围起来的区域的带符号的面积，$x$轴以上的面积为正值，$x$轴以下的面积为负值).
>       + 黎曼积分(Riemann Integral)：
>          + 对于闭区间$[a,b]$，定义$[a,b]$的一个分割为在此区间中取一个有限的点列$a=x_0 \lt x_1 \lt x_2 \lt ... \lt x_N=b.$ 这些点将区间$[a,b]$ 分割为N个子区间$[x_{n-1},x_n],1 \le n \le N$，每个子区间取出一个点$t_n \in [x_{n-1},x_n]$作为代表，函数$f(x)$在这个分割上的黎曼和为$\sum_{n=1}^Nf(t_n)(x_n-x_{n-1}).$
>          + 不同分割的黎曼和不同.当$\lambda=max_{n=1}^N(x_n-x_{n-1})$足够小时，如果所有的黎曼和都趋于某个极限，那么这个极限叫做函数$f(x)$在闭区间$[a,b]$上的黎曼积分.
> + 矩阵微积分(Matrix Calculus)：用矩阵和向量来表示多元函数偏导数. 有2种表示方式：分子布局(Numerator Layout)和分母布局(Denominator Layout).
>    |表示类型|标量关于向量的偏导数|向量关于标量的偏导数|向量关于向量的偏导数|
>    |:------:|:------------------:|:-----------------:|:-----------------:|
>    |$x,y$|标量$y=f(x)\in\mathbb{R}$，<br/>向量$x\in \mathbb{R}^M$.|向量$y=f(x)\in \mathbb{R}^N$，<br/>标量$x\in \mathbb{R}$.|向量$y=f(x)\in \mathbb{R}^N$，<br/>向量$x\in \mathbb{R}^M$.|
>    |分子布局|${\partial y \over \partial x}=[{\partial y \over \partial x_1},...,{\partial y \over \partial x_M}]\in\mathbb{R}^{1\times M}$<br/>($\partial y \over \partial x$为行向量)|${\partial y \over \partial x}=[{\partial y_1 \over \partial x},...,{\partial y_N \over \partial x}]\in \mathbb{R}^{1\times N}$|-|
>    |分母布局<br/>(默认)|${\partial y \over \partial x}=[{\partial y \over \partial x_1},...,{\partial y \over \partial x_M}]^T\in\mathbb{R}^{M\times1}$<br/>($\partial y \over \partial x$为列向量)|${\partial y \over \partial x}=[{\partial y_1 \over \partial x},...,{\partial y_N \over \partial x}]\in \mathbb{R}^{N \times 1}$|${\partial f(x) \over \partial x}=\begin{vmatrix}\partial y_1 \over \partial x_1 & ... & \partial y_N \over \partial x_1\\ \vdots & \ddots & \vdots\\\partial y_1 \over \partial x_M & ... & \partial y_N \over \partial x_M \end{vmatrix}\in \mathbb{R}^{M \times N}$<br/>(函数$f(x)$的Jacobian Matrix的转置.)|
>    + 函数$f(x)$关于$x$的二阶偏导数: 向量$x\in \mathbb{R}^M$，函数$f(x)\in \mathbb{R}$，$H=\nabla^2 f(x)={\partial^2f(x) \over \partial x^2}=\begin{vmatrix}\partial^2 y \over \partial x_1^2 & \cdots & \partial^2 y \over \partial x_1 x_M\\ \vdots & \ddots & \vdots\\ \partial^2 y \over \partial x_M x_1 & \cdots & \partial^2 y \over \partial x_M x_M\end{vmatrix} \in \mathbb{R}^{M \times M}$<br/>(函数$f(x)$的Hessian矩阵).
>    + 导数法则：
>       + 加减法则：若$x \in \mathbb{R}^M,y=f(x) \in \mathbb{R}^N,z=g(x) \in \mathbb{R}^N$，则<br/>${\partial(y+z) \over \partial x}={\partial y \over \partial x}+{\partial z \over \partial x}\in \mathbb{R}^{M \times N}$.
>       + 乘法法则：
>          + 若$x \in \mathbb{R}^M,y=f(x) \in \mathbb{R}^N,z=g(x) \in \mathbb{R}^N$，则<br/>${\partial(y^Tz) \over \partial x}={\partial y \over \partial x}z+{\partial z \over \partial x}y \in \mathbb{R}^M$.
>          + 若$x \in \mathbb{R}^M,y=f(x) \in \mathbb{R}^S,z=g(x) \in \mathbb{R}^T,A \in \mathbb{R}^{S \times T}$且和$x$无关，则<br/>${\partial y^{T}Az \over \partial x}={\partial y \over \partial x}Az+{\partial z \over\partial x}A^Ty \in \mathbb{R}^M$.
>          + 若$x \in \mathbb{R}^M,y=f(x) \in \mathbb{R},z=g(x) \in \mathbb{R}^N$，则<br/>${\partial yz \over \partial x}=y{\partial z \over \partial x}+{\partial y \over \partial x}z^T \in \mathbb{R}^{M \times N}$.
>       + 链式法则(Chain Rule)：
>          + 若$x \in \mathbb{R},y=g(x) \in \mathbb{R}^M,z=f(y) \in \mathbb{R}^N$，则<br/>
>${\partial z \over \partial x}={\partial y \over \partial x}{\partial z \over \partial y}\in \mathbb{R}^{1 \times N}$.
>          + 若$x \in \mathbb{R}^M,y=g(x) \in \mathbb{R}^K,z=f(y) \in \mathbb{R}^N$，则<br/>
>          + 若$x \in \mathbb{R}^{M \times N},y=g(x) \in \mathbb{R}^K,z=f(y) \in \mathbb{R}$，则<br/>
> + 常见函数导数
>    + 向量函数及其导数
>    + Logistic函数
>    + Softmax函数
> ----
> #### Appendix.C 数学优化(Mathematical Optimization)
> + 
>----
> #### Appendix.D 概率论
> + 样本空间：一个随机试验所有可能结果的集合.随机试验中的每个可能结果称为*样本点*.
> + 随机事件/事件：一个被赋予*概率*的事物集合，或者说是样本空间中的一个子集.
> + 概率(Probability)：表示一个*随机事件*发生的可能性大小，为0到1之间的实数.
> + 
>----
> #### Appendix.E 信息论
> +
>----
> Chap.1 
> Chap.2 机器学习概述<br/>


> Chap.4 前馈神经网络

> Chap.5 卷积神经网络
