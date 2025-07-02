import os
import shutil
import re

def organize_by_patient_id(source_folder, dest_folder):
    """
    Move images named like P1_L_CM_MLO.png to dest_folder.
    """
    os.makedirs(dest_folder, exist_ok=True)
    pattern = re.compile(r'^P\d+_[LR]_[A-Z]+_[A-Z]+\.png$', re.IGNORECASE)

    for filename in os.listdir(source_folder):
        if pattern.match(filename):
            src_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(dest_folder, filename)
            shutil.move(src_path, dest_path)
        else:
            print(f"Skipping unexpected file format: {filename}")

def organize_by_view(source_folder, dest_root_folder):
    """
    Organize images by their view (e.g., L_CM_MLO) in separate folders.
    """
    os.makedirs(dest_root_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        parts = filename.split('_')
        if len(parts) >= 4:
            view = '_'.join(parts[1:4])
            view_folder = os.path.join(dest_root_folder, view)
            os.makedirs(view_folder, exist_ok=True)

            src_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(view_folder, filename)

            if not os.path.exists(dest_path):
                os.symlink(os.path.abspath(src_path), dest_path)
        else:
            print(f"Filename doesn't match expected pattern: {filename}")

def parse_filename(filename):
    """
    Returns patient_id and view parsed from filename.
    """
    name = os.path.splitext(filename)[0]
    parts = name.split('_')
    if len(parts) >= 4:
        patient_id = parts[0]
        view = '_'.join(parts[1:4])
        return patient_id, view
    return None, None
