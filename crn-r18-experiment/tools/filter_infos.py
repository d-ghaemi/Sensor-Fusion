import mmcv
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

data_root = Path('data/nuScenes')

def check_exists(info):
    # Check if the main front camera file exists on disk
    cam_filename = info['cam_infos']['CAM_FRONT']['filename']
    return (data_root / cam_filename).exists()

for split in ['train', 'val']:
    pkl_path = data_root / f'nuscenes_infos_{split}.pkl'
    print(f"Loading {split} infos...")
    infos = mmcv.load(pkl_path)
    
    print(f"Filtering {split} using 16 threads...")
    with Pool(16) as p:
        # chunksize=1000 drastically reduces inter-process communication overhead
        exists_mask = list(tqdm(p.imap(check_exists, infos, chunksize=1000), 
                                total=len(infos), desc=f"Filtering {split}"))
    
    filtered_infos = [info for info, exists in zip(infos, exists_mask) if exists]
    mmcv.dump(filtered_infos, pkl_path)
    print(f"✅ {split}: Kept {len(filtered_infos)} / {len(infos)} samples\n")

print("🎉 Info files successfully filtered!")
