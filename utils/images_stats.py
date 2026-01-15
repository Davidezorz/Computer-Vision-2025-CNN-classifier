import os
import collections
from torchvision.datasets import ImageFolder
from PIL import Image

def analyze_subset(root_path, name="Dataset"):
    if not os.path.exists(root_path):
        print(f"Warning: {name} folder not found at {root_path}")
        return

    ds = ImageFolder(root=root_path)
    
   
    class_counts = collections.Counter(ds.targets)                              # Count Classes
    res_counts = collections.Counter()                                          # get (path_to_image, class_index)
    
    for path, _ in ds.samples:
        try:
            with Image.open(path) as img:
                res_counts[img.size] += 1                                       # img.size is (width, height)
        except Exception:
            pass
    
    # Print Class Distribution
    print(f"\n=== {name} Breakdown ===")
    print(f"{'Class Name':<20} | {'Count'}")
    print("-" * 35)
    for i, class_name in enumerate(ds.classes):
        print(f"{class_name:<20} | {class_counts[i]}")

    
    print(f"\n{'Resolution (WxH)':<20} | {'Count'}")                            # Print Resolution Distribution
    print("-" * 35)
    for res, count in res_counts.most_common(10):                               # Print top 10 most common resolutions
        print(f"{str(res):<20} | {count}")
    
    unique_res = len(res_counts)
    if unique_res > 10:
        print(f"... and {unique_res - 10} other variations.")
        
    if unique_res > 1:
        print(f"\n[!] Note: {name} contains images of different sizes.")

if __name__ == "__main__":
    data_root = '.data'
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')

    analyze_subset(train_dir, name="Train")
    analyze_subset(test_dir, name="Test")