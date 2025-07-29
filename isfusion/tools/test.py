# Copyright (c) OpenMMLab. All rights reserved.
"""
这个脚本用于测试 MMDetection3D 模型。

功能:
- 在指定的数据集上对训练好的模型进行测试。
- 支持单 GPU 和多 GPU 测试。
- 支持多种评估指标（如 mAP, recall）。
- 可以将测试结果保存为 pickle 文件。
- 可以将结果格式化以提交到测试服务器。
- 支持可视化测试结果。

用法示例:

# 单 GPU 测试
# python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval ${EVAL_METRICS}
python tools/test.py configs/isfusion/isfusion_0075voxel.py work_dirs/isfusion_0075voxel/latest.pth --eval bbox

# 多 GPU 测试
# ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --eval ${EVAL_METRICS}
./tools/dist_test.sh configs/isfusion/isfusion_0075voxel.py work_dirs/isfusion_0075voxel/latest.pth 8 --eval bbox

参数说明:
- CONFIG_FILE: 模型的配置文件路径。
- CHECKPOINT_FILE: 训练好的模型权重文件路径。
- EVAL_METRICS: 评估指标，例如 'bbox' 或 'mAP'。
"""
import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test, multi_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmcv.utils import ConfigDict

import pickle


def parse_args():
    """
    解析和处理命令行参数。
    该函数使用 argparse 库来定义和解析脚本运行时所需的各种输入参数。
    这些参数允许用户指定配置文件、模型权重、评估选项等。

    返回:
        argparse.Namespace: 包含所有已解析命令行参数的对象。
    """
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    # 必需参数：模型配置文件路径
    parser.add_argument('--config', help='test config file path',
                        default='/home/lishengjie/study/jiahao/bupt_2507/isfusion/configs/isfusion/isfusion_0075voxel.py')
    # 必需参数：模型权重文件路径
    parser.add_argument('--checkpoint', help='checkpoint file',
                        default='/home/lishengjie/study/jiahao/bupt_2507/isfusion/work_dirs/isfusion_0075voxel/epoch8/epoch_3.pth')
    # 可选参数：输出结果的 .pkl 文件路径
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    # 可选参数：保存结果的目录
    parser.add_argument(
        '--result_dir', help='directory where results are saved',
        default='/home/lishengjie/study/jiahao/bupt_2507/isfusion/output/results/')
    parser.add_argument(
        '--bs',
        type=int,
        default=1,
        help='batch size')
    # 可选参数：评估指标
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    # 可选参数：是否实时显示结果
    parser.add_argument('--show', action='store_true', help='show results', default=False)
    # 可选参数：是否显示BEV（鸟瞰图）结果
    parser.add_argument('--show_bev', action='store_true', help='show bev results', default=False)
    parser.add_argument(
        '--show_dir', help='directory where results will be saved',
        default='/home/lishengjie/study/jiahao/bupt_2507/isfusion/output/shows/')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    # 可选参数：随机种子，用于复现结果
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # 可选参数：覆盖配置文件中的某些设置
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    # 可选参数：评估时使用的自定义选项
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    # 可选参数：分布式任务启动器
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    """
    主函数，负责执行整个测试流程。

    流程:
    1. 解析命令行参数。
    2. 加载并合并配置。
    3. 初始化分布式环境（如果使用多GPU）。
    4. 构建数据集和数据加载器。
    5. 构建模型并加载预训练权重。
    6. 执行测试（单GPU或多GPU）。
    7. 处理并评估测试结果。
    """
    # 1. 解析命令行参数
    args = parse_args()

    # 2. 检查必须指定至少一个操作 (保存/评估/格式化/显示结果)
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    # 3. 从文件加载配置
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        # 从命令行参数合并配置
        cfg.merge_from_dict(args.cfg_options)
    # 导入自定义模块
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # 设置 CUDNN benchmark 模式以加速推理
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 测试时不需要预训练模型
    cfg.model.pretrained = None
    # 更新测试时的 batch size
    cfg.data.test.update(dict(samples_per_gpu=args.bs))

    # 为测试数据集做准备
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # 当 batch_size > 1 时, 将 'ImageToTensor' 替换为 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # 4. 初始化分布式环境（如果使用多GPU）
    if args.launcher == 'none':
        distributed = False
        cfg.data.workers_per_gpu = 1
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # 5. 设置随机种子以保证结果可复现
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # 6. 构建数据集
    dataset = build_dataset(cfg.data.test)
    # 构建数据加载器 (dataloader)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # 7. 构建模型并加载权重
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 处理 FP16 设置
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # 从 checkpoint 文件加载模型权重
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # 兼容旧版本 checkpoint，如果其中包含类别信息则加载
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # 加载调色板信息，用于分割任务的可视化
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    # 8. 执行测试
    if not distributed:
        # 单 GPU 测试
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_bev, args.show_dir)
    else:
        # 多 GPU 测试
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    # 9. 主进程 (rank 0) 处理和评估结果
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            # 将输出结果写入 .pkl 文件
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
            # outputs = mmcv.load(args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            # 仅格式化结果，不进行评估
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            # 进行评估
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            if args.result_dir is not None:
                eval_kwargs.update(pklfile_prefix=os.path.dirname(args.result_dir))
            # 打印评估结果
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
