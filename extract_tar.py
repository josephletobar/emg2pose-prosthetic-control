import tarfile
import os

tar_path = "/Volumes/Crucial X9/emg_dataset/emg2pose_dataset.tar"
output_root = "/Volumes/Crucial X9/emg_dataset/by_user"

os.makedirs(output_root, exist_ok=True)

def extract_user_id(filename):
    parts = filename.split("-")
    for i, p in enumerate(parts):
        if p == "cv" and i > 0:
            return parts[i - 1]
    return None

with tarfile.open(tar_path, "r|") as tar:
    for member in tar:
        if not member.isfile():
            continue
        
        name = os.path.basename(member.name)
        user_id = extract_user_id(name)
        
        if user_id is None:
            continue
        
        user_dir = os.path.join(output_root, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        f = tar.extractfile(member)
        if f is None:
            continue
        
        out_path = os.path.join(user_dir, name)
        
        with open(out_path, "wb") as out:
            out.write(f.read())