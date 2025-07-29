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

对此，我们选择了数据集NuScenes, 并且采取开源的代码项目:`IS-Fusion`，来完成这个目标。


## 附录

### 1. 开源仓库

1. [IS-Fusion：](https://github.com/yinjunbo/IS-Fusion)`https://github.com/yinjunbo/IS-Fusion`
2. 

### 2. 参考论文

1. [IS-Fusion：](https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_IS-Fusion_Instance-Scene_Collaborative_Fusion_for_Multimodal_3D_Object_Detection_CVPR_2024_paper.pdf)`https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_IS-Fusion_Instance-Scene_Collaborative_Fusion_for_Multimodal_3D_Object_Detection_CVPR_2024_paper.pdf`