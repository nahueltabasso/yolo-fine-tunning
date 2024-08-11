# This script allow to upload corrected annotations from CVAT to FiftyOne
# %%
import fiftyone as fo
import argparse

# Arguments
parser = argparse.ArgumentParser(description="Script to load a annotations to FiftyOne")
parser.add_argument("--name", type=str, help="Name of dataset", required=True)
parser.add_argument("--anno_key", type=str, help="Annotation key for dataset", required=True)
args = parser.parse_args()

# %%
# Cargamos las anotaciones corregidas en cvat
ds_name = args.name
anno_key = args.anno_key
datasets = fo.load_dataset(ds_name)
print(datasets.list_annotation_runs())

# %%
datasets.load_annotations(anno_key)

# Ejecutar script
# python src/fiftyone/fo_load_annotations.py --name dataset-240728 --anno_key fix_ann