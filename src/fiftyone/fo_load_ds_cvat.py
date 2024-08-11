# This script allow to load a dataset from FiftyOne, filter samples and upload to CVAT
# %%
import fiftyone as fo
import argparse

# Arguments
parser = argparse.ArgumentParser(description="Script to load a dataset to CVAT from FiftyOne")
parser.add_argument("--name", type=str, help="Name of dataset", required=True)
parser.add_argument("--anno_key", type=str, help="Annotation key for dataset", required=True)
parser.add_argument("--project_name", type=str, help="Project name in CVAT", required=True)
args = parser.parse_args()

# %%
ds_name = args.name
anno_key = args.anno_key
project_name = args.project_name
dataset = fo.load_dataset(ds_name)

print(f"Dataset -> {dataset}")

# %%
view = dataset.match_tags(anno_key)
fo.pprint(view.stats(include_media=True))

# %%
view.annotate(
    anno_key,
    project_name=project_name,
    label_field="ground_truth",
    label_type="detections",
    classes=[
        "_",
        "background",
        "credit_card",
    ],
    image_quality=100,
)
print(view.get_annotation_info(anno_key))


# Ejecutar script
# python src/fiftyone/fo_load_ds_cvat.py --name dataset-240728 --anno_key fix_ann --project_name Credit_Card_Dataset_01