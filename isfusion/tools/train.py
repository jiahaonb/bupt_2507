# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp
from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version


def parse_args():
    """
    解析命令行参数。
    该函数定义了所有可用的命令行选项，并返回解析后的参数。
    """
    # 创建一个 ArgumentParser 对象，用于处理命令行参数
    parser = argparse.ArgumentParser(description='训练一个3D检测器')
    # 添加 'config' 参数，这是必须的，用于指定训练配置文件的路径
    parser.add_argument('config', help='训练配置文件的路径')
    # 添加 '--work-dir' 参数，用于指定保存日志和模型的目录
    parser.add_argument('--work-dir', help='保存日志和模型的目录')
    # 添加 '--extra_tag' 参数，为本次实验添加额外的标签
    parser.add_argument('--extra_tag', type=str, default=None, help='本次实验的额外标签')
    # 添加 '--resume-from' 参数，用于指定从哪个检查点文件恢复训练
    parser.add_argument(
        '--resume-from', help='从中恢复的检查点文件')
    # 添加 '--no-validate' 参数，一个标志位，表示在训练期间不进行评估
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='是否在训练期间不评估检查点')
    # 创建一个互斥组，确保 '--gpus' 和 '--gpu-ids' 不能同时使用
    group_gpus = parser.add_mutually_exclusive_group()
    # 添加 '--gpus' 参数，用于指定使用的 GPU 数量（仅适用于非分布式训练）
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='要使用的GPU数量 (仅适用于非分布式训练)')
    # 添加 '--gpu-ids' 参数，用于指定使用的 GPU ID（仅适用于非分布式训练）
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='要使用的GPU的ID (仅适用于非分布式训练)')
    # 添加 '--seed' 参数，用于设置随机种子以保证实验可复现性
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    # 添加 '--deterministic' 参数，一个标志位，用于为CUDNN后端设置确定性选项
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='是否为CUDNN后端设置确定性选项')
    # 添加 '--options' 参数（已弃用），用于覆盖配置文件中的设置
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='覆盖配置文件中的一些设置，键值对格式为 xxx=yyy，'
        '将被合并到配置文件中 (已弃用), 请改用 --cfg-options。')
    # 添加 '--cfg-options' 参数，用于覆盖配置文件中的设置
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='覆盖配置文件中的一些设置，键值对格式为 xxx=yyy，'
        '将被合并到配置文件中。如果被覆盖的值是列表，'
        '应为 key="[a,b]" 或 key=a,b 的形式。'
        '它还允许嵌套的列表/元组值，例如 key="[(a,b),(c,d)]"。'
        '注意引号是必需的，并且不允许有空格。')
    # 添加 '--launcher' 参数，用于指定作业启动器（如 pytorch, slurm）
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='作业启动器')
    # 添加 '--local_rank' 参数，用于分布式训练
    parser.add_argument('--local_rank', type=int, default=0)
    # 添加 '--autoscale-lr' 参数，一个标志位，用于根据GPU数量自动缩放学习率
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='根据GPU数量自动缩放学习率')
    # 解析命令行参数
    args = parser.parse_args()
    # 如果环境变量中没有 'LOCAL_RANK'，则使用命令行参数中的值
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # 检查 '--options' 和 '--cfg-options' 是否同时被指定
    if args.options and args.cfg_options:
        raise ValueError(
            '--options 和 --cfg-options 不能同时指定, '
            '--options 已弃用，推荐使用 --cfg-options')
    # 如果使用了 '--options'，发出警告并将其值赋给 '--cfg-options'
    if args.options:
        warnings.warn('--options 已弃用，推荐使用 --cfg-options')
        args.cfg_options = args.options

    # 返回解析后的参数
    return args


def main():
    """
    主函数，包含了模型训练的整个流程。
    """
                                                                   # 1. 解析命令行参数
    args = parse_args()

    # 2. 从指定的配置文件加载配置
    cfg = Config.fromfile(args.config)
    #    如果通过命令行传递了 cfg-options，则合并到配置中
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # 从字符串列表中导入模块（如果配置中指定了 custom_imports）
    # 这允许用户在配置文件中定义自己的模块，而无需修改框架代码
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # 设置 cudnn_benchmark。当输入大小不变时，设置为 True 可以加速训练
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 3. 确定并设置工作目录 (work_dir)
    #    优先级: 命令行参数 > 配置文件 > 根据配置文件名自动生成
    if args.work_dir is not None:
        # 如果命令行指定了 work_dir，则使用它
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # 如果配置文件中也没有指定 work_dir，则根据配置文件名自动生成一个
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    # 如果指定了 extra_tag，则将其附加到 work_dir 路径中，方便区分实验
    if args.extra_tag is not None:
        cfg.work_dir = osp.join(cfg.work_dir, args.extra_tag)

    # 4. 设置断点恢复和 GPU
    #    如果指定了从断点恢复，则设置 resume_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    #    设置 GPU ID
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # 5. (可选) 自动缩放学习率
    #    如果启用了学习率自动缩放，则根据 GPU 数量线性调整学习率
    if args.autoscale_lr:
        # 应用线性缩放规则 (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # 6. 初始化分布式环境
    if args.launcher == 'none':
        # 非分布式训练
        distributed = False
    else:
        # 分布式训练
        distributed = True
        # 初始化分布式环境 (通信后端等)
        init_dist(args.launcher, **cfg.dist_params)
        # 在分布式训练模式下重新设置 gpu_ids
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # 7. 创建工作目录并初始化日志记录器
    #    创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    #    将最终的配置信息保存到工作目录中，方便复现
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    #    初始化日志记录器
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    #    根据模型类型确定日志记录器名称
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    #    获取根日志记录器
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # 8. 收集并记录元数据 (环境信息、配置、随机种子等)
    meta = dict()
    #    记录环境信息
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('环境信息:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    #    将配置信息存入元数据
    meta['config'] = cfg.pretty_text

    #    记录一些基本信息
    logger.info(f'分布式训练: {distributed}')
    logger.info(f'配置:\n{cfg.pretty_text}')

    # 9. 设置随机种子以保证实验可复现性
    if args.seed is not None:
        logger.info(f'设置随机种子为 {args.seed}, '
                    f'确定性: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    #    将实验名称（配置文件名）存入元数据
    meta['exp_name'] = osp.basename(args.config)

    # 10. 根据配置构建模型
    #     这是核心步骤之一，会递归构建所有模型组件 (backbone, neck, head 等)
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    #     初始化模型权重
    model.init_weights()

    #     记录模型结构到日志
    logger.info(f'模型:\n{model}')
    
    # 11. 根据配置构建数据集
    #     这也是核心步骤，会创建 Dataset 对象，并应用数据处理流水线
    datasets = [build_dataset(cfg.data.train)]
    
    #     如果工作流 (workflow) 中定义了验证阶段，则同样构建验证数据集
    if len(cfg.workflow) == 2:
        # 深拷贝验证集配置以避免修改原始配置
        val_dataset = copy.deepcopy(cfg.data.val)
        # 确保验证集和训练集使用相同的数据处理流程 (pipeline)
        # 这在一些复杂的数据集包装器（如 CBGSDataset）中尤为重要
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # 在验证模式下，test_mode 应为 False，因为我们仍在训练流程中
        # 这不会影响最终的评估指标计算
        val_dataset.test_mode = False
        # 构建并添加验证数据集
        datasets.append(build_dataset(val_dataset))
        
    # 12. 设置检查点 (checkpoint) 配置
    if cfg.checkpoint_config is not None:
        # 在检查点的元数据中保存 mmdet 版本、配置文件内容和类别名称等信息，以便后续使用
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            # 对于分割任务，还需保存调色板
            PALETTE=datasets[0].PALETTE
            if hasattr(datasets[0], 'PALETTE') else None)
            
    # 为模型添加一个 CLASSES 属性，方便后续进行可视化或结果解析
    model.CLASSES = datasets[0].CLASSES
    
    # 13. 调用核心训练函数，启动训练循环
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


# 当脚本作为主程序执行时
if __name__ == '__main__':
    main()
