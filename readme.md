## 深度学习经典论文笔记

本文用于记录阅读深度学习经典论文时个人的一些理解，便于后期回忆，欢迎交流。

---

## 基础架构

### 1.[AlexNet](https://blog.csdn.net/qq_38807688/article/details/84206655)

结构是 (Conv->Relu>Pool->Norm) * 2->(Conv->Relu) * 3->Pool->(Full->Relu->Drop) * 2 ->Full

some keys:

​	1.学习率在整个训练过程中手动调整的。我们遵循的启发式是，当验证误差率在当前学习率下不再降低时，就将学习率除以10。

​	2.卷积or池化时，步长s比核的尺寸size小,这样池化层的输出之间有重叠,提升了特征的丰富性。

​	3.提出了LRN层，[局部响应归一化](https://www.jianshu.com/p/c014f81242e7)，对局部神经元创建了竞争的机制,使得其中响应较大的值变得更大,并抑制反馈较小的。（可以理解为某个点除以它那个h,w处前后depth_radius通道处的平方和）

​	4.显存占用 = 模型显存占用 + batch_size × 每个样本的显存占用，所以节省显存的办法有降低 batch-size、下采样、减少全连接层（一般只留最后一层分类用的全连接层）

​	5.dropout的作用是增加网络的泛化能力，一般用在全连接层（因为参数多）

### 2[.VGGNet](https://zhuanlan.zhihu.com/p/42233779)

结构简洁，整个网络都用同样大小卷积核尺寸和最大池化尺寸，反复堆叠3x3小型卷积核和2x2最大池化层。且拓展性强，迁移其他图片数据泛化性好。

一般结构：nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU

[**亮点1：train image和test image大小是不一样的，是通过在测试时更改最后的全连接层为卷积层实现的，也就是测试时网络是全卷积网络，此时计算方式是不变的，只是会因为test image大小大一些导致最后卷积出来的结果会是 a * a * class_num （而不是 1 * 1 * class_num） ，此时只要对每一个通道（即class_num 的a * a矩阵做一个平均就好，再最后对所有的通道做softmax），对应keys #3**](https://www.zhihu.com/question/53420266)

**亮点2**：最终模型的准确率在 **1. [256;512] train image** (在这个区间随机取一个数字并resize到这个大小进行训练)+ **2.dense eval** (即亮点1，用3个不同的Q取test image进行eval)+ **3.multi-crop**(对于每个Q，都进行随机裁剪并随机翻转) + **4.神经网络融合**（ConvNet Fusion）（融合了D和E，基于文中D和E网络的softmax输出的结果的平均进行预测，因为模型之间可以互补，最后的表现还可以再提升一点）时达到最高

some keys:

​	1.两次3 * 3卷积核相当于一次5 * 5卷积核，但3 * 3参数更少，且通过多级级联可以直接结合非线性层。非线性表现力更强。

​	2.[对于选择softmax分类器还是k个logistics分类器，取决于所有类别之间是否互斥。](https://blog.csdn.net/SZU_Hadooper/article/details/78736765)所有类别之间明显互斥用softmax；所有类别之间不互斥有交叉的情况下最好用k个logistics分类器。

​	3.测试图片的尺寸(Q)不一定要与训练图片的尺寸(S)相同，测试时先将网络转化为全卷积网络，第一个全连接层转为7×7的卷积层，后两个全连接层转化为1×1的卷积层。结果得到的是一个N×N×M的结果，称其为类别分数图，其中M等于类别个数，N的大小取决于输入图像尺寸Q，计算每个类别的最终得分时，将N×N上的值求平均，此时得到1×1×M的结果，此结果即为最终类别分数，这种方法文中称为**密集评估**。

​	4.宽度的增加在一开始对网络的性能提升是有效的。但是，随着宽度的增加，对网络整体的性能其实是开始趋于饱和，并且有下降趋势，因为过多的特征（一个卷积核对应发现一种特征）可能对带来噪声的影响。

### 3.[GoogleNet (Inception)](https://blog.csdn.net/u014061630/article/details/80308245)

**亮点1：有多个辅助分类器**，因为GoogleNet深度很深，为了解决梯度反向传播的问题，将中间的某几层的输出的loss乘以0.3加到最后的cost上，最后推断（inference）时，去除额外的分类器。

亮点2：一个Inception模块由多个子模块组成，如1×1卷积、3×3卷积、5×5卷积和maxpool。

some keys:

​	1.好的网络会把高度相关的节点连在一起，而某层的输出中，在同一个位置但在不同通道的卷积核输出结果相关性极高，一个1×1的卷积核可以很自然的把这些相关性很高，在同一个空间位置，但不同通道的特征结合起来。因此Inception中含有大量的1×1卷积。

​	2.一个Inception模块中其他3×3卷积和5×5卷积还可以继续用来提取特征。

​	3.更大的模型（更多的参数）会导致 计算时间过长、过度拟合训练集、对训练集数量要求过高。

​	4.训练了7个版本的GoogleNet ，它们具有相同的初始化权重，仅在采样方法和随机输入图像的顺序不同，最终的预测结果由这 7个版本共同预测得到，这样泛化能力更强，准确率更高。

​	5.**训练时**，一些模型在较小的crop上训练，一些模型则在大的crop上训练，这些patch的size平均分布在图像区域的8%到100%间，长宽比例在3/4和4/3间，同时有改变亮度、饱和度的操作（photometric distortions），resize时内插方法随机使用bilinear, area, nearest neighbor and cubic, with equal probability。

​	6.**测试时**，先将图片分别resize到256，288，320和352的长/宽，再取这样图片的左，中，右方块（在肖像图片中，我们采用顶部，中心和底部方块），对于取出的方块将采用4个角+中心224×224裁剪图像+resize到224×224，并最终复制一份镜像的版本，这导致每张图像会得到4×3×6×2 = 144的裁剪图像，最终softmax概率在多个裁剪图像上和所有单个分类器上进行平均得到结果。

### [4.ResNet](https://zhuanlan.zhihu.com/p/56961832) 

[作者视频讲解](https://zhuanlan.zhihu.com/p/54072011?utm_source=com.tencent.tim&amp;utm_medium=social&amp;utm_oi=41268663025664)

some keys:

​	1.注意连接求和的时候直接每个对应feature map的对应元素相加即可，注意是先相加再ReLU激活。

​	2.Res18、Res34用的残差块是两个卷积操作，而Res50、Res101、Res152用的残差块是先1×1卷积降维，再3×3卷积提取特征，再1×1卷积升维，我的理解是因为深度深了，需要1×1卷积控制通道大小，而先降维再升维是为了先浓缩特征（通道数少了可以减少卷积的计算量），待卷积提取新的特征后再放大为了和前面的x维度相同，方便相加。

​	3.一个细节：Res50、Res101、Res152每个stage之间的第一个残差块跳远连接时维度不同，需要做一个升维（论文图中的虚线）（对应代码中stride=2部分）。

​	4.**训练时**，和AlexNet、VGGNet一样先每张图片减均值；**数据增强：**利用VGGNet的多尺度处理，从区间[256, 480]随机取一个数 S，将原图resize到短边长度为 S，然后再从这张图随机裁剪出224 x 224大小的图片以及其水平翻转作为模型的输入，除此之外还用了AlexNet的PCA颜色增强（物体的特征不随光照强度和颜色的变化而变）；在卷积之后ReLU之前用了BN；网络初始化方法用[这篇论文](https://arxiv.org/abs/1502.01852)；所有的网络都是从头开始训练；优化使用SGD，batch size = 256，学习率初始值0.1，每当验证集误差开始上升LR就除以10，总迭代次数达到60万次，weight decay (L2正则化前面的参数)= 0.0001，momentum = 0.9；不使用dropout。

​	5.**预测时**，和AlexNet一样进行**TTA**，每张图片有10个裁剪；并且和VGGNet一样采用全卷积形式和多尺度TTA，最后对它们的模型输出值取均值即可。

总结：加了跳远连接后，每个残差块学到的东西会比没有跳远连接学到的东西少，或者可以说学到的东西更精细、精确，这样在深度很深时将学到的各个精细的成分组合起来会达到很好的效果。

### [5.DenseNet](https://zhuanlan.zhihu.com/p/82901676)

由3-4个dense block组成，每个dense block内部由多个Bottleneck layers组成，每个dense block之间是transition layer层，用于减小feature-map的数量和特征的h,w。

some keys:

​	1.与ResNet不同，在特征组合时不通过直接相加，而是用concate来综合特征，增加了输入的变异性并且提高了效率。

​	2.一个dense block中，后面的Bottleneck layers与前面所有的Bottleneck layers都有连接。

​	3.每个Bottleneck layers都是先1×1卷积，压缩通道到4k大小，便于3×3卷积运算，再3×3卷积输出通道大小为k。

​	4.Transition layers做的是BN+1×1Conv+2×2 avg pool下采样，减小feature-map的数量（由θ决定减少到原来的多少，可设为0.5），同时也减少了feature的宽度。

## 神经网络优化

### 1.Adam

含梯度下降、动量、Adagrad、Rmsprop的思想。

### 2.[BatchNorm](https://www.jianshu.com/p/78d3fd2841c9)

batchNorm是在batch上，对**NHW**做归一化，即是将同一个batch中的所有样本的**同一层**特征图抽出来一起求mean和variance。

当网络加深时网络参数的微小变动会被逐渐放大，BatchNorm可以让网络的各层输入保持同一分布有利于提升训练效率。

过程：**先归一化，再去归一化**。归一化即对mini-batch上的数据缩放到服从N(0,1)的正态分布![\widehat{x}](https://math.jianshu.com/math?formula=%5Cwidehat%7Bx%7D)，去归一化实际上是一个缩放平移操作![y^{(k)}=\gamma^{(k)}x^{(k)}+\beta^{(k)}](https://math.jianshu.com/math?formula=y%5E%7B(k)%7D%3D%5Cgamma%5E%7B(k)%7Dx%5E%7B(k)%7D%2B%5Cbeta%5E%7B(k)%7D)，其中![\beta](https://math.jianshu.com/math?formula=%5Cbeta)和![\gamma](https://math.jianshu.com/math?formula=%5Cgamma)都是可学习的，因为如果把网络每一层的特征都归一化会影响网络的学习能力。其中，γ与β是可学习的**大小为C**的参数向量（C为输入大小)。

也可以理解为归一化操作限制了网络的学习能力，去归一化操作引入了新的可学习参数，把丢失的学习能力又补了上来。

关于**测试阶段，要使用固定的mean和var**，主流的自动微分框架会在训练时用**滑动平均**的方式统计mean和var，当你使用测试模式的时候，框架会自动载入统计好的mean与var。

缺点：

​	1.需要较大的batch size

​	2.对于可变长度的训练，如RNN，不太适用

​	3.训练阶段需要保存均值和方差，以便测试时使用

### 3.LayerNorm

LN其实就是对每个样本数据作标准化，所以LN不受batch size影响，LayerNorm计算的均值和方差只用于一个样本中，不用于后面的样本数据，LayerNorm中也存在![\gamma](https://private.codecogs.com/gif.latex?%5Cgamma)和![\beta](https://private.codecogs.com/gif.latex?%5Cbeta)可学习参数，并且![\gamma](https://private.codecogs.com/gif.latex?%5Cgamma)和![\beta](https://private.codecogs.com/gif.latex?%5Cbeta)是在特征维度进行，而不是在Batch维度。

### 4.GroupNorm

GroupNorm将channel分组，即是将batch中的单个样本的G层特征图抽出来分别求mean和variance，与batch size无关。输入通道被分成num_groups组，每个组包含num_channels / num_groups个通道。每组的均值和标准差分开计算。

当batch size较小时(小于16时)，使用GroupNorm方法效果更好。

当GroupNorm中num_groups的数量是1的时候, 是与上面的LayerNorm是等价的。

```python
#BN
#num_features： C来自期待的输入大小(N,C,H,W)，一般就是通道大小
torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)

#LN
#normalized_shape (int or list or torch.Size)： 来自期待输入大小的输入形状
#若input.size为(2,3,2,2) 则normalized_shape为input.size()[1:]，即torch.Size([3, 2, 2])
torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True)

#GN
#num_channels：输入通道数
#num_groups：要分成的组别数
torch.nn.GroupNormnum_channels(num_groups, num_channels, eps=1e-05, affine=True)
```

[【总结】](https://www.cnblogs.com/wanghui-garcia/p/10877700.html)

​	1.在batch size较大，数据分布比较接近时，BN的效果更好，但LN的应用范围更广（小mini-batch场景、动态网络场景和 RNN，特别是自然语言处理领域）。 此外，LN 不需要保存 mini-batch 的均值和方差，节省了额外的存储空间。

​	2.BN 的转换是针对单个神经元可训练的——不同神经元的输入经过再平移和再缩放后分布在不同的区间，而 LN 对于一整层的神经元训练得到同一个转换——所有的输入都在同一个区间范围内。如果不同输入特征不属于相似的类别（比如颜色和大小），那么 LN 的处理可能会降低模型的表达能力。

### [5.GAN](https://zhuanlan.zhihu.com/p/83476792)

原始GAN是由一个生成器G和一个判别器D组成的，生成器的目的就是将随机输入的高斯噪声映射成图像（"假图"），判别器则是判断输入图像是否来自生成器的概率，即判断输入图像是否为假图的概率，最优的状态就是“假图”在判别器D中的输出是0.5，即D不知道这幅图是真是假。

损失函数：

![[公式]](https://www.zhihu.com/equation?tex=+%5Cunderset%7BG%7D%7B%5Cmathop%7B%5Cmin+%7D%7D%5C%2C%5Cunderset%7BD%7D%7B%5Cmathop%7B%5Cmax+%7D%7D%5C%2CV%28D%2CG%29%3D%7B%7B%5Cmathbb%7BE%7D%7D%7Bx%5Csim+%7B%7Bp%7D%7Bdata%7D%7D%28x%29%7D%7D%5B%5Clog+D%28x%29%5D%2B%7B%7B%5Cmathbb%7BE%7D%7D%7Bz%5Csim+%7B%7Bp%7D%7Bdata%7D%7D%28z%29%7D%7D%5B%5Clog+%281-D%28G%28z%29%29%29%5D%5Ctag1+)

其中， ![[公式]](https://www.zhihu.com/equation?tex=G) 代表生成器， ![[公式]](https://www.zhihu.com/equation?tex=D) 代表判别器， ![[公式]](https://www.zhihu.com/equation?tex=x) 代表真实数据， ![[公式]](https://www.zhihu.com/equation?tex=p_%7Bdata%7D) 代表真实数据概率密度分布， ![[公式]](https://www.zhihu.com/equation?tex=z) 代表了随机输入数据，该数据是随机高斯噪声。

从判别器 ![[公式]](https://www.zhihu.com/equation?tex=D) 角度来看判别器 ![[公式]](https://www.zhihu.com/equation?tex=D) 希望能尽可能区分真实样本 ![[公式]](https://www.zhihu.com/equation?tex=x) 和虚假样本 ![[公式]](https://www.zhihu.com/equation?tex=G%28z%29) ，因此 ![[公式]](https://www.zhihu.com/equation?tex=D%28x%29) 必须尽可能大， ![[公式]](https://www.zhihu.com/equation?tex=D%28G%28z%29%29) 尽可能小， 也就是 ![[公式]](https://www.zhihu.com/equation?tex=V%28D%2CG%29) 整体尽可能大。从生成器 ![[公式]](https://www.zhihu.com/equation?tex=G) 的角度来看，生成器 ![[公式]](https://www.zhihu.com/equation?tex=G) 希望自己生成的虚假数据 ![[公式]](https://www.zhihu.com/equation?tex=G%28z%29) 可以尽可能骗过判别器 ![[公式]](https://www.zhihu.com/equation?tex=D) ，也就是希望 ![[公式]](https://www.zhihu.com/equation?tex=D%28G%28z%29%29) 尽可能大，也就是 ![[公式]](https://www.zhihu.com/equation?tex=V%28D%2CG%29) 整体尽可能小。GAN的两个模块在训练相互对抗，最后达到全局最优。

那么如何训练？

![image-20201017201334397](C:\Users\金子\AppData\Roaming\Typora\typora-user-images\image-20201017201334397.png)

即先取m个噪声数据映射为噪声分布，和m个正样本映射成真实数据分布，固定生成器G，利用梯度上升训练判别器D使得上述V（D,G）最大化，这一步进行k次。然后取m个噪声数据映射为噪声分布，固定判别器D，利用梯度下降训练生成器G。上述过程不断迭代。

## 目标检测、分割

【说明】目标检测的主流算法主要分为两类：（1）**two-stage方法**，如R-CNN系算法，其主要思路是先通过启发式方法（selective search）或者CNN网络（RPN)产生一系列稀疏的候选框，然后对这些候选框进行分类与回归，two-stage方法的优势是准确度高；（2）**one-stage方法**，如Yolo和SSD，其主要思路是均匀地在图片的不同位置进行密集抽样，抽样时可以采用不同尺度和长宽比，然后利用CNN提取特征后直接进行分类与回归，整个过程只需要一步，所以其优势是速度快，但是均匀的密集采样的一个重要缺点是训练比较困难，这主要是因为正样本与负样本（背景）极其不均衡（参见[Focal Loss](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1708.02002)），导致模型准确度稍低。

### [1.RCNN](https://blog.csdn.net/shenxiaolu1984/article/details/51066975)

算法步骤如下：

- 一张图像生成1K~2K个**候选区域**
- 对每个候选区域，使用深度网络**提取特征**
- 特征送入每一类的SVM **分类器**，判别是否属于该类
- 使用回归器**精细修正**候选框位置

### [2.Fast RCNN](https://blog.csdn.net/shenxiaolu1984/article/details/51036677)

算法步骤如下：

- 先将整张图像输入网络中提取特征，在邻接时，才加入候选框信息（候选区域的前几层特征不需要再重复计算，在末尾的少数几层处理每个候选框，解决了RCNN速度慢的问题），并进行ROI池化，得到固定长度的feature vector 特征表示，将得到的feature 一分为二，一个输入到proposal 分类的全连接，另一个输入到用于bounding box regression的全连接中去。
- 把类别判断和位置精调统一用深度网络实现，两个损失函数相加，不再需要额外存储。

注：roi_pool层将每个候选区域均匀分成M×N块，对每块进行max pooling。将特征图上大小不一的候选区域转变为大小统一的数据，送入下一层。

测试时，利用窗口得分分别对每一类物体进行非极大值抑制剔除重叠建议框，最终得到每个类别中回归修正后的得分最高的窗口。

### [3.Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)

Faster R-CNN中引入Region Proposal Network(RPN)替代Selective Search，同时引入anchor box应对目标形状的变化问题（anchor就是位置和大小固定的box，可以理解成事先设置好的固定的proposal）

算法步骤如下：

- 对整张图片输进CNN，得到feature map
- feature map输入到RPN，得到候选框的特征信息
- 对候选框中提取出的特征进行[ROI POOLing](https://blog.csdn.net/AUTO1993/article/details/78514071)，使用分类器判别是否属于一个特定类 
- 对于属于某一类别的候选框，用回归器进一步调整其位置

关于RPN部分的详解，可以参考[这篇文章](https://blog.csdn.net/u011746554/article/details/74999010)。

![img](https://img-blog.csdn.net/20170324121024882)

### 4.YOLO v1

思想是一个单一的神经网络在一次评估中直接从完整图像预测边界框和类概率，使用回归的方法。

即 将一幅图像分成SxS个网格，每个网络需要预测B个BBox，每个bounding box要预测(x, y, w, h)和confidence共5个值，每个网格还要预测一个类别信息，记为C类，则输出就是S x S x (5*B+C)的一个tensor。

训练时：

​	1.一幅图像卷积+全连接后得到S x S x (5*B+C)的一个tensor，每个点的通道含(5*B+C)个数据。

​	2.对于包含object的点，做类别预测，并对与ground truth box IOU最大的那个box（称为负责的box）做confidence error进行惩罚（将confidence push到1）（其他的box不做confidence error）和位置回归（更大的loss weight），这样这个置信度最大box的confidence会越来越大，位置会越来越准确。

​	3.对于不含object的网格中的box也对confidence error（只不过是将其confidence push到0），但是前面需要加一个更小的weight

测试时：

​	1.每个网格取预测的confidence最大的那个bounding box

​	2.对保留的boxes进行NMS处理，就得到最终的检测结果

缺点：

​	1.虽然每个格子可以预测B个bounding box，但是最终只选择只选择IOU最高的bounding box作为物体检测输出，即每个格子最多只预测出一个物体。

​	2.loss函数中，大物体IOU误差和小物体IOU误差对网络训练中loss贡献值接近（虽然采用求平方根方式，但没有根本解决问题）。

### [5.YOLO v2](https://blog.csdn.net/u014380165/article/details/77961414)

![img](https://img-blog.csdn.net/20180910193802403?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xldmlvcGt1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

相对于YOLO v1的改进点：

- BN：为每个卷积层都添加了BN层
- High Resolution Classifier：YOLOv2将预训练分成两步，先用224\*224的输入从头开始训练网络，大概160个epoch（表示将所有训练数据循环跑160次），然后再将输入调整到448\*448，再训练10个epoch。注意这两步都是在ImageNet数据集上操作。最后再在检测的数据集上fine-tuning，也就是detection的时候用448\*448的图像作为输入就可以顺利过渡了。作者的实验表明这样可以提高几乎4%的MAP
- Convolutional With Anchor Boxes：卷积后得到13*13的feature map，每个点又有对应的固定的Anchor Boxes
- new network：使卷积后的feature map是奇数的，这样中心点只有一个，因为有一个预测框在中心的概率很大，这样回归时速度更快
- Dimension priors：作者采用k-means的方式对训练集的bounding boxes做聚类，试图找到合适的anchor box
- Location prediction：作者没有采用直接预测offset的方法，还是沿用了YOLO v1算法中直接预测相对于grid cell的坐标位置的方式。前面提到网络在最后一个卷积层输出13*13大小的feature map，然后每个cell预测5个bounding box，然后每个bounding box预测5个值：tx，ty，tw，th和to（这里的to类似YOLO v1中的confidence）。看下图，tx和ty经过sigmoid函数处理后范围在0到1之间，这样的归一化处理也使得模型训练更加稳定；cx和cy表示一个cell和图像左上角的横纵距离；pw和ph表示预设的bounding box的宽高，这样bx和by就是cx和cy这个cell附近的anchor来预测tx和ty得到的结果。

![这里写图片描述](https://img-blog.csdn.net/20170913081748999?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

- passthrough：将前面一层的26 * 26的feature map和本层的13 * 13的feature map进行连接，有点像ResNet，因为13 * 13的feature map对于比较小的object不友好，所以加上26 * 26的，即网格更小，预测更准。
- Multi-Scale：：就是在训练时输入图像的size是动态变化的，即在训练网络时，每训练10个batch，网络就会随机选择另一种size的输入，作者采用32的倍数作为输入的size，具体来讲文中作者采用从{320,352,…,608}的输入尺寸。这样会使模型更加robust。

### [6.YOLO v3](https://blog.csdn.net/leviopku/article/details/82660381)

在v2的基础上，有以下改进：

- 用了DarkNet-53作为backbone提取特征，同时也提供了DarkNet-19和tiny-DarkNet，在速度和准确率上有一个自主的权衡。
- 有两个cancat拼接过程，用的是前面特征层和后面特征层的上采样（插值的方法使图片变大）
- 聚类后得到9个cachor box，分别为大、中、小各3个，对应与y1、y2、y3这三个feature map，即y1负责预测大的检测框，y2、y3以此类推。
- 每个feature map通道数为255，是因为对于COCO类别而言，有80个种类，每个box应该对每个种类都输出一个概率，同时每个box需要有(x, y, w, h, confidence)五个基本参数，则255对应于3个box，每个box 5+80个预测值，3 * （5+80）= 255，
- 每个网格有3个anchor box，在predict之前会对每个anchor box做一个评分，只取有必要的部分进行logistic回归。我的理解是和目标object的IOU很大或者很小，这样可以把confidence push到1或0，但是文中的意思YOLO v3中每个网格好像最多只取1个anchor box（评分最高的那个box）进行回归
- 损失函数部分，除了w, h的损失函数依然采用平方误差之外，其他部分的损失函数用的是二值交叉熵

![img](https://img-blog.csdn.net/2018100917221176?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xldmlvcGt1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### [7.YOLO v4](https://zhuanlan.zhihu.com/p/150127712)

相对于v3来说就是结合了各种tricks，并以YOLO V3为基础进行改进，YOLO V4在保证速度的同时，大幅提高模型的检测精度，有以下几点改进：

- 相较于YOLO V3的DarkNet53，YOLO V4用了CSPDarkNet53
- 相较于YOLO V3的FPN,YOLO V4用了SPP+PAN
- CutMix数据增强和马赛克（Mosaic）数据增强
- DropBlock正则化
- [损失函数](https://zhuanlan.zhihu.com/p/159209199)部分，分类损失和置信度损失没变，但是bounding box regression损失改变了，v3中x，y和h，w是分开归回的，但是x，y，h，w他们是有一定关系的，在v4中用了一种称为CIOU的损失函数来进行位置回归，CIOU综合考虑到了真实框和预测框的**重叠面积**（越大越好，即1-IOU（A,B）越小越好），**中心点距离**，**长宽比**，定义如下：

![[公式]](https://www.zhihu.com/equation?tex=L_%7BCIOU%7D%3D1-IOU%28A%2CB%29%2B%5Crho%5E%7B2%7D%28A_%7Bctr%7D%2CB_%7Bctr%7D%29%2Fc%5E%7B2%7D%2B%5Calpha.v)

还有很多改进（例如自对抗训练Adversarial Training等等）...这里就不举例了，可以参考原文

![img](https://pic3.zhimg.com/v2-4d60d4d8319e0213491bb52a179e152e_r.jpg)

### [8.SSD](https://zhuanlan.zhihu.com/p/33544892)

全名Single Shot MultiBox Detector，即one-stage方法，MultiBox指SSD是多框预测，采用CNN提取的不同尺度的特征图来来直接进行检测。

步骤如图所示：![img](https://pic1.zhimg.com/v2-a43295a3e146008b2131b160eec09cd4_r.jpg)

注：

- 前面feature map的size较大，提取特征范围小，所以用来预测小的目标，后面的feature map size小，提取特征范围大，用来预测大的目标，不同大小的feature map预设的anchor box大小不同（大小是根据一个公式算出来的，比例由预设值得到），且每个网格单元预设的anchor box数量也不同（有4个有6个）
- 令 ![[公式]](https://www.zhihu.com/equation?tex=n_k) 为特征图所采用的anchor box数目，那么类别置信度需要的卷积核数量为 ![[公式]](https://www.zhihu.com/equation?tex=n_k%5Ctimes+c) （c包含背景，即真正的类别数只有c-1个），而边界框位置需要的卷积核数量为 ![[公式]](https://www.zhihu.com/equation?tex=n_k%5Ctimes+4) 。由于每个先验框都会预测一个边界框，所以SSD300一共可以预测 ![[公式]](https://www.zhihu.com/equation?tex=38%5Ctimes38%5Ctimes4%2B19%5Ctimes19%5Ctimes6%2B10%5Ctimes10%5Ctimes6%2B5%5Ctimes5%5Ctimes6%2B3%5Ctimes3%5Ctimes4%2B1%5Ctimes1%5Ctimes4%3D8732) 个边界框，这是一个相当庞大的数字，所以说SSD本质上是密集采样
- 训练时，首先要确定训练图片中的ground truth（真实目标）与哪个先验框来进行匹配，与之匹配的先验框所对应的边界框将负责预测它。在Yolo中，ground truth的中心落在哪个单元格，该单元格中与其IOU最大的边界框负责预测它。但是在SSD中却完全不一样，SSD的先验框与ground truth的匹配原则主要有两点。首先，**第一个原则**，对于图片中每个ground truth，找到与其IOU最大的先验框，该先验框与其匹配，这样，可以保证每个ground truth一定与某个先验框匹配。通常称与ground truth匹配的先验框为正样本，反之，若一个先验框没有与任何ground truth进行匹配，那么该先验框只能与背景匹配，就是负样本。一个图片中ground truth是非常少的， 而先验框却很多，如果仅按第一个原则匹配，很多先验框会是负样本，正负样本极其不平衡，所以需要第二个原则。**第二个原则**是：对于剩余的未匹配先验框，若某个ground truth的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BIOU%7D) 大于某个阈值（一般是0.5），那么该先验框也与这个ground truth进行匹配。这意味着某个ground truth可能与多个先验框匹配。第二个原则一定在第一个原则之后进行，仔细考虑一下这种情况，如果某个ground truth所对应最大 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BIOU%7D) 小于阈值，并且所匹配的先验框却与另外一个ground truth的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BIOU%7D) 大于阈值，那么该先验框应该匹配谁，答案应该是前者，**首先要确保某个ground truth一定有一个先验框与之匹配**。但是，这种情况我觉得基本上是不存在的。由于先验框很多，某个ground truth的最大 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BIOU%7D) 肯定大于阈值，所以可能只实施第二个原则既可以了。
- 尽管一个ground truth可以与多个先验框匹配，但是ground truth相对先验框还是太少了，所以负样本相对正样本会很多。为了保证正负样本尽量平衡，SSD采用了hard negative mining，就是对负样本进行抽样，抽样时按照置信度误差（预测背景的置信度越小，误差越大）进行降序排列，选取误差的较大的top-k作为训练的负样本，以保证正负样本比例接近1:3

### 9.RetinaNet

全文针对现有one-stage目标检测模型中前景(positive)和背景(negatives)类别的不平衡问题，提出了一种叫做[**Focal Loss**](https://blog.csdn.net/wfei101/article/details/79477303)的损失函数：FL(p<sub>t</sub>) = -α<sub>t</sub>(1-p<sub>t</sub>)<sup>γ</sup>log(p<sub>t</sub>)，而普通的交叉熵为CE(p<sub>t</sub>) = -log(p<sub>t</sub>)

​	1.第一个改进点为α<sub>t</sub>，用于解决**类别数量不平衡**问题，对于属于少数类别的样本，增大α即可，这样类别数少的样本，在计算loss时，loss会更大一些，loss大学习会相对容易一些。

​	2.第二个改进点为(1-p<sub>t</sub>)<sup>γ</sup>，用于解决**难/易分样本不平衡**问题，一旦乘上了该权重，样本越易分，pt越大，(1-p<sub>t</sub>)<sup>γ</sup>则越小，loss就会很小，贡献的对loss的下降就会很小，反之，样本越难分，预测的pt就会很小，此时计算出来的loss会相对大一些，则它对loss下降的贡献就会大一些，模型会更加注意这种类别的学习。在实验中，发现 γ = 2 ， α = 0.25 的取值组合效果最好。

### 10.FPN

虽然不同层的特征图尺度已经不同，形似金字塔结构，但是前后层之间由于不同深度的影响，语义信息差距太大，主要是高分辨率的低层特征很难有代表性的检测能力，所以SSD虽然利用了低层的特征，但是在小目标检测上还有待提升。

而FPN把低分辨率、高语义信息的高层特征 和 高分辨率、低语义信息的低层特征进行自上而下的侧边连接，使得所有尺度下的特征都有丰富的语义信息。

![这里写图片描述](https://img-blog.csdn.net/20170707153508827?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSmVzc2VfTXg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

把FPN应用于Faster RCNN时，不同层融合后的feature map产生不同的anchor box，那么这些anchor box选择哪一层的feature map做ROI POOLing合适呢？注意不是哪层产生的anchor box就用于哪层的feature map，而是经过一个公式计算得到的，公式如下：

![这里写图片描述](https://img-blog.csdn.net/20170117220352603?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSmVzc2VfTXg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

224是ImageNet的标准输入，k0是基准值，设置为5，代表P5层的输出（原图大小就用P5层），w和h是ROI区域（就是变形后的anchor box，region of interest）的长和宽，假设ROI是112 * 112的大小，那么k = k0-1 = 5-1 = 4，意味着该ROI应该使用P4的特征层。k值会做取整处理，防止结果不是整数。

### 11.FCN

![img](https://img-blog.csdn.net/20160508234037674)

蓝色：卷积

绿色：maxpool

灰色：crop

橙色：反卷积操作

黄色：相加

当我们想从2\*2的特征图映射到32\*32的特征图时，我们只要填入相应的参数（包括filter，padding，stride），这些参数与特征图32\*32直接卷积成2\*2的特征图参数**一致**。

### [12.Mask RCNN](https://zhuanlan.zhihu.com/p/37998710)

Mask RCNN = ResNet-FPN+Fast RCNN+mask，也就是加上FPN的Faster RCNN再加上mask

![img](https://pic1.zhimg.com/v2-18b0db72ed142c8208c0644c8b5a8090_r.jpg)

前面提取的特征参照Faster RCNN和FPN那两篇文章，没什么变化，需要注意RPN提取的不同大小的anchor box不一定是去前面那个feature map切割的，是根据一个公式计算出来的，因为大的anchor box最好去靠后的feature map切割，而Mask RCNN这篇paper提出了**两个创新点**来解决实例分割：

​	1.**用ROI Align代替ROI pooling**：普通的Faster RCNN中因为要对anchor box取整再进行划分pool，会造成“不匹配问题”（misalignment），使得ROI不准确，（别看不到1像素的偏移很小，但是在后面分辨率低的feature map上1像素的偏移会造成很大的偏差）。而ROI Align方法取消整数化操作，保留了小数，使用双线性插值的方法获得坐标为浮点数的像素点上的图像数值。具体操作可以看上面这篇[文章](https://zhuanlan.zhihu.com/p/37998710)。

​	2.多了mask分支：注意多出来的mask分支是对ROI Align提取的固定大小的feature map进行的，也就是说只在目标检测的基础上进行mask，或者说最后的实例涂色是只在那个方框里进行的。

![img](https://pic1.zhimg.com/80/v2-a500524ae104ae4eaf9a929befe2ba0c_720w.jpg)

最后的损失是：![[公式]](https://www.zhihu.com/equation?tex=L%3DL_%7Bcls%7D%2BL_%7Bbox%7D%2BL_%7Bmask%7D)，其中 ![[公式]](https://www.zhihu.com/equation?tex=L_%7Bmask%7D)：

假设一共有K个类别，则mask分割分支的输出维度是 m * m * k , 对于 m * m 中的每个点，都会输出K个二值Mask（**每个类别使用sigmoid输出**）。需要注意的是，训练计算loss的时候，并不是每个类别的sigmoid输出都计算二值交叉熵损失，而是该像素属于哪个类，哪个类的sigmoid输出才要计算损失。并且在测试的时候，是通过分类分支预测的类别来选择相应的mask预测。这样，mask预测和分类预测就彻底解耦了。

这与FCN方法是不同的，FCN是对每个像素进行多类别softmax分类，然后计算交叉熵损失，很明显，这种做法是会造成类间竞争的，而每个类别使用sigmoid输出并计算二值损失，可以避免类间竞争。实验表明，通过这种方法，可以较好地提升性能。

### [13.FCOS](https://zhuanlan.zhihu.com/p/63868458)

FCOS: Fully Convolutional One-Stage Object Detection，是基于FCN的逐像素目标检测算法，是anchor free的，核心思想就是预测输入图像中每个点所属的目标类别和目标框。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190605224049230.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQzODAxNjU=,size_16,color_FFFFFF,t_70)

一些关键点：

- feature map上的位置(x,y)可以通过下面这个公式换算成输入图像的位置，其中s表示缩放比例，这样就方便计算特征图上每个点的分类和回归目标

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190605224034429.jpg)

- Classification部分，输入图像上每个点的类别标签根据这个点是否在标注框内来确定，在标注框外的点就是负样本，类别设置为0
- Regression部分，FCOS的回归部分预测4个值(l, t, r, b)，分别表示目标框内某个点离框的左边、上边、右边、下边的距离，因为这4个值都是正的，所以为了保证回归支路的输出结果都是正，回归支路的输出会通过exp()函数再输出
- 为了解决某个像素点同时出现在两个目标框中时，那么这个像素点该属于哪个类的问题，FCOS在特征提取部分参照了FPN思想做了特征融合，不同的feature map对于不同大小的目标框，这也是假设两个重叠的目标框大小有差异这个方法才会生效，若两个目标框大小差不多，则选择一个最合适的即可。
- Center-ness部分，因为作者发现误差产生主要是因为部分误检框离真实框的中心点距离较大，因此作者设计了这样一个分支（可以理解为代表一个权重，称为“中心度”，对应的损失函数如下，可以看到离中心点越远的点权重越小），测试时，将预测的中心度与相应的分类分数相乘，计算最终得分(用于对检测到的边界框进行排序)。因此，中心度可以降低远离对象中心的边界框的权重。这些低质量边界框很可能被最终的非最大抑制（NMS）过程滤除，从而显着提高了检测性能。

![[公式]](https://www.zhihu.com/equation?tex=cenerness%5E%7B%2A%7D%3D%5Csqrt%7B%5Cfrac%7B%5Cmin+%5Cleft%28l%5E%7B%2A%7D%2C+r%5E%7B%2A%7D%5Cright%29%7D%7B%5Cmax+%5Cleft%28l%5E%7B%2A%7D%2C+r%5E%7B%2A%7D%5Cright%29%7D+%5Ctimes+%5Cfrac%7B%5Cmin+%5Cleft%28t%5E%7B%2A%7D%2C+b%5E%7B%2A%7D%5Cright%29%7D%7B%5Cmax+%5Cleft%28t%5E%7B%2A%7D%2C+b%5E%7B%2A%7D%5Cright%29%7D%7D)

### [14.CenterNet](https://zhuanlan.zhihu.com/p/66048276)

CenterNet将目标作为一个点（目标框的中心点），采用关键点估计来找到中心点，并回归到其他目标属性。此时仅仅将图像传入全卷积网络，得到一个热力图，热力图峰值点即中心点，每个特征图峰值点的对应位置回归了目标的宽高信息，因此此方法不需要后期进行NMS处理，因为一个目标框只对应热力图中的一个峰值点。

最终训练时的损失函数分为3部分，分别是L<sub>k</sub> (类别)，L<sub>size</sub> (宽、高)，L<sub>off</sub> (中心点偏差)

![img](https://img-blog.csdnimg.cn/20190417194613626.png)

- L<sub>k</sub> 类别（热力图），将原始图片输入到网络，输出的热力图是![\hat{Y}\epsilon [0,1]^{\frac{W}{R}\times \frac{H}{R}\times C}](https://private.codecogs.com/gif.latex?%5Chat%7BY%7D%5Cepsilon%20%5B0%2C1%5D%5E%7B%5Cfrac%7BW%7D%7BR%7D%5Ctimes%20%5Cfrac%7BH%7D%7BR%7D%5Ctimes%20C%7D) ，C为类别数，![\hat{Y}_{x,y,c}=1](https://private.codecogs.com/gif.latex?%5Chat%7BY%7D_%7Bx%2Cy%2Cc%7D%3D1) 表示检测到的关键点；![\hat{Y}_{x,y,c}=0](https://private.codecogs.com/gif.latex?%5Chat%7BY%7D_%7Bx%2Cy%2Cc%7D%3D0) 表示背景 ，那么热力图的GT怎么得到（为了后续训练）？是根据原始图片中的关键点通过高斯核分散到热力图中得到的， 对于原始图片的关键点 c ,其位置为 ![p \epsilon R^{2}](https://private.codecogs.com/gif.latex?p%20%5Cepsilon%20R%5E%7B2%7D) ，计算得到低分辨率（经过下采样）上对应的关键点 ![\tilde{p}=\left \lfloor \frac{p}{R} \right \rfloor](https://private.codecogs.com/gif.latex?%5Ctilde%7Bp%7D%3D%5Cleft%20%5Clfloor%20%5Cfrac%7Bp%7D%7BR%7D%20%5Cright%20%5Crfloor) . 我们将 GT 关键点通过高斯核![img](https://img-blog.csdnimg.cn/20190417191729651.png)分散到热力图![img](https://img-blog.csdnimg.cn/20190417191817652.png) 上 ，其中![img](https://img-blog.csdnimg.cn/20190417191851264.png) 是一个与目标大小(也就是w和h)相关的标准差。（可以看上面那个[链接](https://zhuanlan.zhihu.com/p/66048276)，写的很详细）
- L<sub>off</sub> 位置偏差，因为图像下采样时，GT的关键点会因数据是离散的而产生偏差（除了后产生小数然后取了整），所有类别 c 共享同个偏移预测
- L<sub>size</sub> 目标框的高、宽回归，和对应的GT产生L1损失

[15.DETR](https://zhuanlan.zhihu.com/p/266069794)









## 自然语言处理、时序建模

### 1.LSTM

![image-20201024200611652](C:\Users\金子\AppData\Roaming\Typora\typora-user-images\image-20201024200611652.png)

### 2.Seq2Seq

![image-20201024200703347](C:\Users\金子\AppData\Roaming\Typora\typora-user-images\image-20201024200703347.png)

### 3.Transformer

Encoder部分重点看[这篇文章](https://blog.csdn.net/yujianmin1990/article/details/85221271)

Decoder部分重点看[这篇文章](https://zhuanlan.zhihu.com/p/34781297)

建议上面两篇都看

