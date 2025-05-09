import os

def CreateYamlFile(dataset_dir):
    images_path = os.path.join(dataset_dir, "images")
    labels_path = os.path.join(dataset_dir, "labels")
    yaml_path = f"{os.path.dirname(dataset_dir)}.yaml"
    with open(yaml_path, 'w') as y:
        y.write(f"""
path: {os.path.abspath(dataset_dir)}
train: {os.path.relpath(images_path, dataset_dir)}
val: {os.path.relpath(images_path, dataset_dir)}
names:
    0: makk_7
    1: makk_8
    2: makk_9
    3: makk_10
    4: makk_also
    5: makk_felso
    6: makk_kiraly
    7: makk_asz
    8: sziv_7
    9: sziv_8
    10: sziv_9
    11: sziv_10
    12: sziv_also
    13: sziv_felso
    14: sziv_kiraly
    15: sziv_asz
    16: tok_7
    17: tok_8
    18: tok_9
    19: tok_10
    20: tok_also
    21: tok_felso
    22: tok_kiraly
    23: tok_asz
    24: zold_7
    25: zold_8
    26: zold_9
    27: zold_10
    28: zold_also
    29: zold_felso
    30: zold_kiraly
    31: zold_asz
""")        
    print(f"{yaml_path} file created.")
    return yaml_path
