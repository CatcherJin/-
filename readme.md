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
- 对候选框中提取出的特征，使用分类器判别是否属于一个特定类 
- 对于属于某一类别的候选框，用回归器进一步调整其位置

关于RPN部分的详解，可以参考[这篇文章](https://blog.csdn.net/u011746554/article/details/74999010)。

![img](https://img-blog.csdn.net/20170324121024882)