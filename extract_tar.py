import tarfile
import os

tar_path = "/Volumes/Crucial X9/emg_dataset/emg2pose_dataset.tar"
output_root = "/Volumes/Crucial X9/emg_dataset/by_user"

os.makedirs(output_root, exist_ok=True)
print(f"Output directory ready: {output_root}")

def extract_user_id(filename):
    parts = filename.split("-")
    for i, p in enumerate(parts):
        if p == "cv" and i > 0:
            return parts[i - 1]
    return None

with tarfile.open(tar_path, "r|") as tar:
    print(f"Opened tar file: {tar_path}")
    
    count = 0
    skipped = 0

    for member in tar:
        if not member.isfile():
            continue
        
        name = os.path.basename(member.name)
        user_id = extract_user_id(name)
        
        if user_id is None:
            continue
        
        user_dir = os.path.join(output_root, user_id)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir, exist_ok=True)
            print(f"Created directory for user {user_id}")
        
        f = tar.extractfile(member)
        if f is None:
            print(f"Failed to extract: {name}")
            continue
        
        out_path = os.path.join(user_dir, name)

        if os.path.exists(out_path):
            skipped += 1
            print(f"Skipping duplicate: {out_path}")
            continue
        
        with open(out_path, "wb") as out:
            out.write(f.read())
        
        count += 1
        
        if count % 100 == 0:
            print(f"Processed {count} files... (Skipped {skipped})")
    
    print(f"Done. Total saved: {count}, skipped duplicates: {skipped}")