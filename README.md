### 简介

首届**“一带一路”国际大数据竞赛** Rank 33 方案，得分 0.74731。

使用 Pytorch 框架，参考了[baseline1](<https://github.com/czczup/UrbanRegionFunctionClassification>)、[baseline2](https://github.com/ABadCandy/BaiDuBigData19-URFC)、[baseline3](https://github.com/ZHOUXINWEN/2019Baidu-XJTU_URFC)（感谢！），主要步骤为：

1. 删除图片中大部分黑和绝大部分白的数据

2. 将visit数据矩阵化（7x26x24），然后结合图像数据（进行了随机翻转、旋转、模糊、形变等增强操作）构建网络进行训练

3. 构建了两种网络进行训练：

   - 将visit数据reshape为1x56x78并pad为1x100x100，接下来将其与图片数据concatenate，组成4x100x100的数据作为输入。backbone网络为se_resnext。
   - 双模态网络，参考baseline1
   - 首先将图片数据经过两个Bottleneck结构转化为32x26x26，visit数据经过一层卷积转化为32x26x26，随后将两者进行concatenate，进而输入到随后的网络种。backbone网络为se_resnext、unet。

   实测表明，前两种网络性能差异不大，甚至第一种看似暴力的方法反而略有优势（第一种网络在验证集精度上高了1个百分点），第3种网络验证集精度较之前二者低了1-2个百分点，但是其在训练前期过程中具有完美的抗过拟合能力。

4. 多种模型训练得到之后，进行**集成**以及**TTA**操作，并利用CatBoost进行最后的Blending操作（可提升5个千分点左右）

### 环境

Windows 10 平台，CPU Xeon E5-2630 v4，GPU RTX2080Ti，内存128G

Python 3.7 + Pytorch 1.1.0

### 主要文件说明

urfc_option.py：相关参数定义

preprocess.py：预处理步骤，包含第1、2步

train.py：进行训练

evaluate.py：进行验证以及Blending

predict.py：进行预测并输出结果

cnn.py：网络的定义