# 使用说明

本仓库为：

题目2：基于深度学习的点云目标检测算法实现的仓库。

其中，您可以根据`isfusion/docker/`下的dockerfile，构建对应的docker镜像，进行复现。也可以通过`isfusion/docs`下的文档，完成环境的安装（conda/python）

我们的训练数据和模型放在`isfusion/work_dirs`目录下。由于时间和算力有限，我们训练次数较少。pth文件即为每次训练的权重。其中对于`tf_logs`，我们可以使用tensorboard进行可视化。评估结果放在`isfusion/output/pkls/pkl`目录下。

通过实验报告的`实验流程`部分，我们可以使用其中的命令行完成网络构建、点云数据处理、训练、评估、可视化等流程。

# 提要内容：

数据集：为nuscenes-mini数据集。由于过大，没有添加。官网：[Nuscenes](https://www.nuscenes.org/) `https://www.nuscenes.org/`

项目代码：本仓库，或附件压缩包；`https://github.com/jiahaonb/bupt_2507`
训练日志即权重：代码`isfusion/work_dirs`目录下；
结果报告：`report/report.md`；附件pdf版本；
展示内容：通过结果报告可视化部分进行可视化。




























































































