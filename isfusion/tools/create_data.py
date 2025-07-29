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


def lyft_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """
    准备 Lyft 数据集的相关数据。
    主要任务是创建记录基本信息的 '.pkl' 文件。
    尽管 Lyft 数据集通常不使用真值数据库和2D标注，但也可以像 nuScenes 一样生成它们。

    Args:
        root_path (str): 数据集根目录路径。
        info_prefix (str): info 文件的前缀。
        version (str): 数据集版本。
        max_sweeps (int, optional): 每个样本最多包含的连续雷达扫描帧数。默认为 10。
    """
    # --- 创建 Lyft 的 info 文件 ---
    lyft_converter.create_lyft_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """
    为 ScanNet 室内场景数据集准备 info 文件。

    Args:
        root_path (str): 数据集根目录路径。
        info_prefix (str): info 文件的前缀。
        out_dir (str): 生成的 info 文件的输出目录。
        workers (int): 使用的线程数。
    """
    # 调用 indoor_converter 来创建 ScanNet 的 .pkl 文件
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def s3dis_data_prep(root_path, info_prefix, out_dir, workers):
    """
    为 S3DIS 室内场景数据集准备 info 文件。

    Args:
        root_path (str): 数据集根目录路径。
        info_prefix (str): info 文件的前缀。
        out_dir (str): 生成的 info 文件的输出目录。
        workers (int): 使用的线程数。
    """
    # 调用 indoor_converter 来创建 S3DIS 的 .pkl 文件
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def sunrgbd_data_prep(root_path, info_prefix, out_dir, workers):
    """
    为 SUN RGB-D 室内场景数据集准备 info 文件。

    Args:
        root_path (str): 数据集根目录路径。
        info_prefix (str): info 文件的前缀。
        out_dir (str): 生成的 info 文件的输出目录。
        workers (int): 使用的线程数。
    """
    # 调用 indoor_converter 来创建 SUN RGB-D 的 .pkl 文件
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=5):
    """
    为 Waymo Open Dataset 准备数据。
    这个过程比较复杂，通常包括：
    1. 将 Waymo 原生格式转换为 KITTI 格式，因为很多模型是基于 KITTI 格式开发的。
    2. 基于转换后的 KITTI 格式数据生成 info 文件。
    3. 创建真值数据库。

    Args:
        root_path (str): 数据集根目录路径。
        info_prefix (str): info 文件的前缀。
        out_dir (str): 生成的 info 文件的输出目录。
        workers (int): 使用的线程数。
        max_sweeps (int): 每个样本最多包含的连续雷达扫描帧数。默认为 5。
    """
    from tools.data_converter import waymo_converter as waymo
    splits = ['training', 'validation', 'testing']
    if version != 'v1.0-mini': # v1.0-mini 版本有不同的处理方式
        # --- 将 Waymo 原生格式转换为 KITTI 格式 ---
        for i, split in enumerate(splits):
            load_dir = osp.join(root_path, 'waymo_format', split)
            if split == 'validation':
                # 在 KITTI 格式中，Waymo 的验证集通常也放在 training 文件夹下
                save_dir = osp.join(out_dir, 'kitti_format', 'training')
            else:
                save_dir = osp.join(out_dir, 'kitti_format', split)
            converter = waymo.Waymo2KITTI(
                load_dir,
                save_dir,
                prefix=str(i),
                workers=workers,
                test_mode=(split == 'test'))
            converter.convert()
            
    # --- 基于 KITTI 格式生成 Waymo 的 info 文件 ---
    out_dir = osp.join(out_dir, 'kitti_format')
    version_postfix = '' if version == 'v1.0' else '_mini'
    info_prefix += version_postfix
    # kitti.create_waymo_info_file(out_dir, info_prefix, max_sweeps=max_sweeps, \
    #     version=version)
    
    # --- 创建真值数据库 ---
    create_groundtruth_database(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False,
        with_bbox=True)


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