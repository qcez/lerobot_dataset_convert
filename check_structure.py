import h5py
import os

train_dir = "/home/agilex/cobot_magic/collect_data/data/folding_clothes_251/"
hdf5_file = os.path.join(train_dir, "episode_0.hdf5")  
with h5py.File(hdf5_file, 'r') as f:
    def print_structure(name, obj):
        print(name)
    
    print("HDF5 文件结构:")
    f.visititems(print_structure)
    
    print("\n顶级键:")
    print(list(f.keys()))
    
    if 'action' in f:
        print(f"action shape: {f['action'].shape}")
    
    if 'observations/images/cam_high' in f:
        print(f"cam_high images count: {len(f['observations/images/cam_high'])}")