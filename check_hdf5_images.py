import h5py
import numpy as np

# HDF5 文件路径
file_path = "/home/agilex/cobot_magic/collect_data/data/folding_clothes_251/episode_0.hdf5"

print("检查 HDF5 文件图像编码格式\n" + "="*60)

with h5py.File(file_path, 'r') as f:
    # 递归打印所有键
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"\n数据集: {name}")
            print(f"  形状 (shape): {obj.shape}")
            print(f"  数据类型 (dtype): {obj.dtype}")
            print(f"  大小 (size): {obj.size}")
            
            # 如果是图像数据，检查详细信息
            if 'images' in name:
                print(f"  数组维度: {len(obj.shape)}")
                if len(obj.shape) >= 3:
                    print(f"  可能的格式: 时间序列图像")
                    if len(obj.shape) == 4:
                        print(f"  帧数: {obj.shape[0]}")
                        print(f"  高度: {obj.shape[1]}")
                        print(f"  宽度: {obj.shape[2]}")
                        print(f"  通道数: {obj.shape[3]}")
                    
                # 读取第一个元素查看值范围
                if obj.size > 0:
                    try:
                        if len(obj.shape) == 1:
                            sample = obj[0] if hasattr(obj[0], '__len__') else obj[:]
                        elif len(obj.shape) == 2:
                            sample = obj[0, 0]
                        elif len(obj.shape) == 3:
                            sample = obj[0, :, :]
                        elif len(obj.shape) == 4:
                            sample = obj[0, 0, 0, :]
                        else:
                            sample = None
                        
                        if sample is not None and hasattr(sample, '__len__'):
                            sample_array = np.array(sample)
                            print(f"  样本数据类型: {sample_array.dtype}")
                            print(f"  样本数据范围: [{sample_array.min()}, {sample_array.max()}]")
                            print(f"  样本形状: {sample_array.shape}")
                    except Exception as e:
                        print(f"  无法读取样本: {e}")
        elif isinstance(obj, h5py.Group):
            print(f"\n组 (Group): {name}")
    
    f.visititems(print_structure)
    
    # 专门检查图像路径
    print("\n\n" + "="*60)
    print("专门检查图像数据:\n")
    
    cameras = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    for cam in cameras:
        cam_key = f'observations/images/{cam}'
        if cam_key in f:
            data = f[cam_key]
            print(f"\n{cam}:")
            print(f"  完整路径: {cam_key}")
            print(f"  形状: {data.shape}")
            print(f"  数据类型: {data.dtype}")
            
            # 读取第一帧图像
            if len(data.shape) == 4:
                first_frame = data[0]
                print(f"  第一帧形状: {first_frame.shape}")
                print(f"  第一帧数据类型: {first_frame.dtype}")
                print(f"  第一帧数值范围: [{first_frame.min()}, {first_frame.max()}]")
                print(f"  第一帧均值: {first_frame.mean():.2f}")
                
                # 判断编码格式
                if first_frame.dtype == np.uint8:
                    if first_frame.max() <= 255 and first_frame.min() >= 0:
                        print(f"  ✓ 编码格式: 原始像素值 (uint8, 0-255)")
                        print(f"  ✓ 图像尺寸: {first_frame.shape[0]}x{first_frame.shape[1]}")
                        if len(first_frame.shape) == 3:
                            if first_frame.shape[2] == 3:
                                print(f"  ✓ 颜色通道: RGB (3通道)")
                            elif first_frame.shape[2] == 1:
                                print(f"  ✓ 颜色通道: 灰度 (1通道)")
                            else:
                                print(f"  ? 颜色通道: 未知 ({first_frame.shape[2]}通道)")
                elif first_frame.dtype in [np.float32, np.float64]:
                    print(f"  ✓ 编码格式: 浮点数像素值 ({first_frame.dtype})")
                    if first_frame.max() <= 1.0:
                        print(f"  ✓ 数值范围: [0, 1] (归一化)")
                    else:
                        print(f"  ? 数值范围: [{first_frame.min()}, {first_frame.max()}]")
                else:
                    print(f"  ? 未知格式: {first_frame.dtype}")
            elif len(data.shape) == 2 and data.dtype == object:
                print(f"  ! 可能是压缩的图像字节流")
                print(f"  图像数量: {data.shape[0]}")
                first_image_bytes = data[0]
                print(f"  第一张图像字节长度: {len(first_image_bytes)}")
                print(f"  前16字节: {bytes(first_image_bytes[:16]).hex()}")
                
                # 检测图像格式
                header = bytes(first_image_bytes[:16])
                if header[:2] == b'\xff\xd8':
                    print(f"  ✓ 图像格式: JPEG")
                elif header[:8] == b'\x89PNG\r\n\x1a\n':
                    print(f"  ✓ 图像格式: PNG")
                else:
                    print(f"  ? 图像格式: 未知 (可能是原始数据或自定义格式)")
        else:
            print(f"\n{cam}: 不存在")

print("\n" + "="*60)
