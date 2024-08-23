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
        # "_",
        "card_number",
        "cardholder",
        "expiry_date",
        "payment_network"
    ],
    image_quality=100,
    overwrite=True
)
print(view.get_annotation_info(anno_key))


# print(dataset.list_annotation_runs())
# dataset.delete_annotation_run(anno_key)
# print(dataset.list_annotation_runs())
