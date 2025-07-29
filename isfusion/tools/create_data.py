# Copyright (c) OpenMMLab. All rights reserved.

import argparse
from os import path as osp

from tools.data_converter import indoor_converter as indoor
from tools.data_converter import kitti_converter as kitti
from tools.data_converter import lyft_converter as lyft_converter
from tools.data_converter import nuscenes_converter as nuscenes_converter
from tools.data_converter.create_gt_database import create_groundtruth_database


def kitti_data_prep(root_path, info_prefix, version, out_dir):
    """
    准备 KITTI 数据集的相关数据。
    相关数据包括：
    1. 记录基本信息的 '.pkl' 文件。
    2. 2D 标注信息。
    3. 用于数据增强的真值数据库（Ground Truth Database）。

    Args:
        root_path (str): 数据集的根目录路径。
        info_prefix (str): 生成的 info 文件的前缀。
        version (str): 数据集版本（例如 'v1.0'）。
        out_dir (str): 真值数据库 info 文件的输出目录。
    """
    # --- 创建 KITTI 数据集的 info 文件 ---
    # 这两个函数会生成 kitti_infos_train.pkl, kitti_infos_val.pkl 等文件
    # kitti.create_kitti_info_file(root_path, info_prefix)
    # kitti.create_reduced_point_cloud(root_path, info_prefix) # 为Velodyne点创建降采样点云

    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(root_path,
                                  f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
    
    # --- 导出 2D 标注 ---
    # 从 .pkl 文件中提取 2D 标注信息，通常用于多模态模型的训练或可视化
    # kitti.export_2d_annotation(root_path, info_train_path)
    # kitti.export_2d_annotation(root_path, info_val_path)
    # kitti.export_2d_annotation(root_path, info_trainval_path)
    # kitti.export_2d_annotation(root_path, info_test_path)

    # --- 创建真值数据库 (Ground Truth Database) ---
    # 这个数据库用于 "GT-Sampling" 数据增强，即将训练集中的真实物体样本“粘贴”到其他场景中
    create_groundtruth_database(
        'KittiDataset', # 数据集名称
        root_path,      # 数据集根目录
        info_prefix,    # info 文件前缀
        f'{out_dir}/{info_prefix}_infos_train.pkl', # 训练集 info 文件路径
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'), # 是否带有 mask
        with_bbox=True) # 是否带有 bbox


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """
    准备 nuScenes 数据集的相关数据。
    
    相关数据包括：
    1. 记录基本信息的 '.pkl' 文件（包含多帧点云 sweep 信息）。
    2. 2D 标注信息。
    3. 用于数据增强的真值数据库。

    Args:
        root_path (str): 数据集根目录路径。
        info_prefix (str): info 文件的前缀。
        version (str): 数据集版本 (例如 'v1.0-trainval', 'v1.0-test', 'v1.0-mini')。
        dataset_name (str): 数据集类名 (例如 'NuScenesDataset')。
        out_dir (str): 真值数据库 info 文件的输出目录。
        max_sweeps (int): 每个样本最多包含的连续雷达扫描帧数。默认为 10。
    """
    # --- 创建 nuScenes 的 info 文件 ---
    # 这个函数会处理 nuScenes 的复杂数据结构，生成包含多帧信息的 .pkl 文件
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        # 对于测试集，只导出 2D 标注用于可能的评估或可视化
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_test_path, version=version)
        return

    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    # 导出训练和验证集的 2D 标注
    # nuscenes_converter.export_2d_annotation(
    #     root_path, info_train_path, version=version)
    # nuscenes_converter.export_2d_annotation(
    #     root_path, info_val_path, version=version)
    
    # --- 为训练集创建真值数据库 ---
    create_groundtruth_database(dataset_name, root_path, info_prefix,
                                f'{out_dir}/{info_prefix}_infos_train.pkl',
                                with_bbox=True)
    
    # --- 添加的代码开始 ---
    print('Start creating KITTI-style dataset...')

    # 检查目标文件夹是否存在，如果不存在则创建
    kitti_format_out_dir = osp.join(out_dir, 'kitti_format')
    if not osp.exists(kitti_format_out_dir):
        from pathlib import Path
        Path(kitti_format_out_dir).mkdir(parents=True, exist_ok=True)

    # 调用核心转换函数，将 .pkl 信息转换为 KITTI 格式
    # 注意：这里的函数名可能是 _create_kitti_format_infos 或者其他类似名称
    # 您需要根据您的 nuscenes_converter.py 文件确认
    # 但在较新的版本中，这个功能通常集成在 create_nuscenes_infos 函数内部的一个参数控制
    # 我们可以先尝试调用一个独立的函数
    if version != 'v1.0-test':
        # 转换训练集
        info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        nuscenes_converter.export_to_kitti(
            info_path=info_train_path,
            root_path=root_path,
            out_dir=kitti_format_out_dir,
            info_prefix=info_prefix,
            dataset_version=version
        )
        # 转换验证集
        info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
        nuscenes_converter.export_to_kitti(
            info_path=info_val_path,
            root_path=root_path,
            out_dir=kitti_format_out_dir,
            info_prefix=info_prefix,
            dataset_version=version
        )
    # --- 添加的代码结束 ---



# --- 命令行参数解析 ---
# 创建一个 ArgumentParser 对象来处理命令行输入
parser = argparse.ArgumentParser(description='数据转换器参数解析器')
# 'dataset' 参数：必须提供，指定要处理的数据集名称，例如 'kitti'
parser.add_argument('dataset', metavar='kitti', help='数据集的名称')
# '--root-path' 参数：数据集的根目录路径
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='指定数据集的根目录路径')
# '--version' 参数：数据集的版本，对 kitti 不是必需的
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='指定数据集版本，kitti无需此参数')
# '--max-sweeps' 参数：每个样本使用的雷达扫描帧数
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='指定每个样本的雷达扫描次数')
# '--out-dir' 参数：输出目录，用于存放生成的 .pkl 文件
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='info pkl文件的输出目录')
# '--extra-tag' 参数：info 文件名的额外标签（前缀）
parser.add_argument('--extra-tag', type=str, default='kitti')
# '--workers' 参数：处理数据时使用的线程数
parser.add_argument(
    '--workers', type=int, default=4, help='要使用的线程数')
# 解析所有定义的命令行参数
args = parser.parse_args()


# --- 主程序入口 ---
if __name__ == '__main__':
    # 根据用户指定的 'dataset' 参数，调用相应的数据准备函数
    if args.dataset == 'kitti':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir)
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        # 完整版 nuScenes 数据集，分别处理 trainval 和 test 集
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        # Mini 版 nuScenes 数据集
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'lyft':
        # Lyft 数据集，分别处理 train 和 test
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'waymo':
        # Waymo 数据集
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 's3dis':
        s3dis_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)