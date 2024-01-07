from single_image_dataset.affectnet import AffectNetDataset
from single_image_dataset.raf_db import RAFDBDataset
from config import EMOTION_CLASSIFICATION, AFFECTNET, INSTRUCTION_DATA_PATH, RAF_DB
import os
import json

if __name__ == "__main__":
    affectnet = AffectNetDataset(
        root_path=EMOTION_CLASSIFICATION,
        dataset_path=AFFECTNET,
        template_path="dataset/template/emotion_classification.json",
    )
    affectnet_dataset = affectnet.build()

    raf_db = RAFDBDataset(
        root_path=EMOTION_CLASSIFICATION,
        dataset_path=RAF_DB,
        template_path="dataset/template/emotion_classification.json",
    )
    raf_db_dataset = raf_db.build()

    dataset = affectnet_dataset + raf_db_dataset

    # Save the dataset
    if not os.path.exists(INSTRUCTION_DATA_PATH):
        os.makedirs(INSTRUCTION_DATA_PATH)
        # Write the updated dataset to the JSON file
    with open(INSTRUCTION_DATA_PATH, "w") as file:
        json.dump(dataset, file, indent=2)
