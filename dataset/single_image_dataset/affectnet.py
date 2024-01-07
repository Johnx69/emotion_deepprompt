import os
import json
import random


class AffectNetDataset:
    def __init__(
        self,
        root_path="data/emotion_classification_dataset",
        dataset_path="AffectNet/AffectNet_v2",
        template_path="template/emotion_classification.json",
    ):
        self.root_path = root_path
        self.dataset_path = dataset_path
        self.template_path = template_path
        self.emotion_labels = [
            "Neutral",
            "Happy",
            "Sad",
            "Surprise",
            "Fear",
            "Disgust",
            "Angry",
            "Contempt",
        ]

    def build(self):
        dataset = []
        # Load the template
        question_list = json.load(open(self.template_path, "r"))
        # Read the AffectNet_v2_list_train_emot.txt file
        list_train_emot_path = os.path.join(
            self.root_path, self.dataset_path, "AffectNet_v2_list_train_emot.txt"
        )

        with open(list_train_emot_path, "r") as file:
            lines = file.readlines()

        # Process each line and append to the dataset
        for line in lines:
            image_path, label = line.strip().split()
            _, _, _, i, imagename = image_path.split("/")
            image_id = imagename.split(".")[0]

            # Randomly select a question from the template
            question = random.choice(question_list)

            # Create the entry for the dataset
            entry = {
                "id": image_id,
                "image": os.path.join(self.root_path, image_path),
                "conversations": [
                    {"from": "human", "value": question},
                    {
                        "from": "gpt",
                        "value": f"The emotion of this image is {self.emotion_labels[int(label)].lower()}",
                    },
                ],
            }

            # Append to the dataset
            dataset.append(entry)
        return dataset


# # Example usage:
# affectnet_dataset = AffectNetDataset()
# entries = affectnet_dataset.build()
# print(f"Data has been written to {affectnet_dataset.json_output_path}")
