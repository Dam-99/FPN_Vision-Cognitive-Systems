import fiftyone as fo

dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path="data/train2017",
    labels_path="data/annotations/instances_train2017.json",
    # label_types="detections",
    include_id=True,
    persistent=True,
)

session = fo.launch_app(dataset=dataset)

session.wait()