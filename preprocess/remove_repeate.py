def remove_duplicate_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    unique_lines = list(set(lines))  # 去重（仅适用于完全相同的行）
    with open(label_path, 'w') as f:
        f.writelines(unique_lines)


# 调用示例
label_file = "../../VisDrone2019-YOLO/labels/train/9999945_00000_d_0000114.txt"
remove_duplicate_labels(label_file)