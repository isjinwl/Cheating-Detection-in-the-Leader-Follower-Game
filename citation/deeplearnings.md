## Convolutional Neural Network(CNN)
----
>**Neural Network and Deep Learning(邱锡鹏，2020)**[[Book]](https://nndl.github.io/)<br/>
> #### Appendix.A 线性代数
> + N维向量
>    + 标量(Scalar):是一个实数，只有大小，没有方向。e.g.,*a,b,c,...*
>    + 向量(Vector):是一组实数构成的有序数组，有大小，有方向.e.g.,N维向量 ***a***=[$a_1$, $a_2$, ..., $a_N$]
>    + 向量空间(Vector Space or Linear Space): 向量组成的集合，且满足*向量加法*和*标量乘法*（两者可视为向量空间的条件）.
>       + 欧氏空间(Euclidean Space): 一个欧氏空间通常表示为$\mathbb{R}^N$，N为空间维度(Dimension).
>          + 欧氏空间中的*向量加法*: <br/>
***a***+***b***=[$a_1, a_2, ..., a_N$]+[$b_1, b_2, ..., b_N$]=[$a_1+b_1,a_2+b_2,...,a_N+b_N$]
>          + 欧氏空间中的*标量乘法*：<br/>
*c* $\cdot$ ***a*** = *c* $\cdot$ [$a_1, a_2, ..., a_N$]=[*c*$a_1$, *c*$a_2$, ..., *c*$a_N$]. 其中*a,b,c* $\in \mathbb{R}$ 属于标量.
>       + 线性子空间：向量空间的一个子集，也满足*向量加法*和*标量乘法*.
>       + 线性相关(linearly dependent): 如果向量空间中的一个向量可以用有限个其他向量的线性组合表示，则称线性相关.<br/>或线性空间$\mathbb{v}$中的**M**个向量{$v_1, v_2,...,v_M$}，如果$\lambda_1 v_1+\lambda_2 v_2+...+\lambda_M v_M=0$存在非零解$\lambda_1,\lambda_2,...,\lambda_M$（标量），则向量组{$v_1, v_2,...,v_M$}是线性相关的；否则向量组{$v_1, v_2,...,v_M$}是线性无关的(linearly independent).
>        + 基向量：N维向量空间$\mathcal{V}$的基(Base)$\mathcal{B}$={$e_1,e_2,...,e_N$}是$\mathbb{v}$的有限子集，其元素之间线性无关。向量空间$\mathbb{v}$的所有向量都可以按唯一的方式表达为$\mathcal{B}$中向量的线性组合，即对于任意 $v \in \mathcal{V}$存在一组标量{$\lambda_1,\lambda_2,...,\lambda_N$}，使得$v=\lambda_1e_1+\lambda_2e_2+...+\lambda_Ne_N$. 且如果基向量是有序的，则($\lambda_1,\lambda_2,...,\lambda_N$)称为向量$v$关于基$\mathcal{B}$的坐标.如果基向量是Standard Basis,($\lambda_1,\lambda_2,...,\lambda_N$)则可称为向量$v$的笛卡尔坐标(Cartesian Coordinate).
>        + 内积(Inner Product|Dot Product|Scalar Product): 一个N维向量空间中的2个向量***a***和***b***的内积为$<a,b>=\sum_{n-1}^Na_nb_n$(基向量相同,cos$\theta$=1).
>        + 正交(Orthogonal)：如果同一个向量空间中2个向量的内积为0，则这两个向量正交；如果向量空间A中的一个向量$v$与子空间$\mathcal{U}$的每个向量正交，那么向量$v$与子空间$\mathcal{U}$正交.
> + 范式(Norm): 
> #### Appendix.B 微积分
> #### Appendix.C 数学优化
> #### Appendix.D 概率论
> #### Appendix.E 信息论
> Chap.1 
> Chap.2 机器学习概述<br/>


> Chap.4 前馈神经网络

> Chap.5 卷积神经网络
