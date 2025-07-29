# 基于深度学习的点云目标检测算法实现

> 本任务旨在实现一个基于神经网络的三维点云目标检测系统，要求能够对原始或预处理后的三维点云数据进行目标识别与检测，输出目标的类别及其三维包围框信息。

任务目标：

1. 数据准备与预处理

   - 1.1 熟悉并解析点云数据格式（如 `.bin`、`.pcd`、`.ply` 等）；
   - 1.2 实现点云数据的归一化、裁剪、投影、采样等预处理操作；
   - 1.3 数据集划分合理，标注信息可用于训练监督。
2. 模型设计与训练：

    - 2.1 构建或改进点云目标检测网络；
    - 2.2 实现数据加载模块、网络前向传播、损失函数与训练过程
    - 2.3 模型需能输出目标类别与其三维边界框（center + size + orientation）。

3. 模型评估与测试：

    - 在测试集上进行模型性能评估；
    - 可视化检测结果（支持显示检测框与真实框对比）；
    - 评估指标包括但不限于：3D mAP（Mean Average Precision）、IoU、Precision、Recall 等。

4. 结果可视化与模型管理：

    - 支持训练过程中的损失曲线、mAP 曲线可视化；
    - 支持保存和加载训练权重；
    - 支持调用模型进行测试推理并输出结果。

对此，我们选择了数据集NuScenes, 并且采取开源的代码项目:`IS-Fusion`，来完成这个目标。由于数据集过大（300GB+），我们最终选择的是`NuScenes-mini`数据集(3GB+).

## 1. 查看方法

我们得知，`IS-Fusion`是CVPR'2024 论文，论文地址是`https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_IS-Fusion_Instance-Scene_Collaborative_Fusion_for_Multimodal_3D_Object_Detection_CVPR_2024_paper.pdf`。该工作同时对实例级和场景级多模态上下文进行建模，以增强 3D 检测性能。我们可以查看对应的论文，了解其原理、方法和贡献，并且针对性地分析其问题所在。

首先，我们经过调研，了解到了之前处理3D点云目标跟踪的方法，包括BEV-Fusion，

## 2. 实验流程

我们从这里开始，进行`IS-Fusion`的实验配置部分。包括环境配置、数据集准备、模型训练、模型评估、结果可视化等。我们会分析方法的实验过程，运行过程，从而建立起对方法的优缺点的评价。之后，就可以针对其中的问题提出必要的改进。

### 2.1 环境配置

我们的`IS-Fusion`是基于`PyTorch`的大框架的，基于 torch 1.10.1、mmdet 2.14.0、mmcv 1.4.0 和 mmdet3d 0.16.0。所以第一步是创建环境并且安装对应的包文件。

```bash
conda create -n isfusion python=3.7
conda activate isfusion
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
```

由于MMDetection3D 依赖于 MMDetection，因此 mmcv-full 是必需的。mmcv有其自己的安装工具:mim。我们可以通过这个安装。mim 是 OpenMMLab 项目的包管理工具。

```bash
pip install -U openmim
mim install mmcv
```

这里mim会直接选择最合适我们的mmcv-full版本。除此之外，我们也可以按照对应的cuda版本以及pytorch版本来选择其他的mmcv-full版本。

接下来就是mmdetection和MMSegmentation，也是我们的基础。

```bash
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

下面是最重要的一部分，也就是安装mmdetection3d.这个使我们可以对点云等数据进行处理。这个是在Github的发布，所以我们需要下载源代码并且通过setup来进行安装。

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v -e .
```

最后需要安装torchEX,这个是在`mmdet3d/ops/TorchEx`目录下。

```bash
cd mmdet3d/ops/TorchEx
pip install -v -e .
```

完成以上的安装，我们再使用pip安装一些基础的库，在`ISFusion/requirements.txt`中。一般来说，完成上面的步骤，就可以全部安装完这里面的包了。

```bash
pip install -r requirements.txt
```

最后，我们可以通过pip freeze > new_requirements.txt来保存我们的包，或者通过`conda env export > environment.yaml`来保存我们的环境文件，便于之后进行复现。其实原仓库里面， 已经有了docker的配置文件，我们可以直接在另一台机器中使用docker复现这个场景。

### 2.2 数据集准备

完成了环境的配置，就可以开始进行数据集的配置了。我们按照官网的配置，链接了我们的数据集：由于我们准备的是nuscenes-mini数据集，但是要求要放在`mmdetection3d/data/nuscenes`目录下。所以我们创建软链接：

```bash
ln -s /home/lishengjie/data/nuscenes-mini /home/lishengjie/study/jiahao/bupt_2507/ISFusion/data/nuscenes
```

然后对数据集进行配置：

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini
```

我们可以观察到，这里的

### 2.3 模型训练

我们可以看到，在`tools/train.py`中，有对应的训练代码。然后我们可以调用`tools/run-nus.sh`，来针对性地对这个nuscenes数据集进行处理，并且开始训练的过程。\

```bash
bash tools/run-nus.sh 'tag'
```

后面设置‘tag’，方便我们查看log以及训练数据等。





## 附录

### 1. 开源仓库

1. [IS-Fusion：](https://github.com/yinjunbo/IS-Fusion)`https://github.com/yinjunbo/IS-Fusion`

2. 

### 2. 参考论文

1. [IS-Fusion：](https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_IS-Fusion_Instance-Scene_Collaborative_Fusion_for_Multimodal_3D_Object_Detection_CVPR_2024_paper.pdf)`https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_IS-Fusion_Instance-Scene_Collaborative_Fusion_for_Multimodal_3D_Object_Detection_CVPR_2024_paper.pdf`