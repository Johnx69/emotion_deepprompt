import os
import json
import random
import scipy.io


class RAFDBDataset:
    def __init__(
        self,
        root_path="data/emotion_classification_dataset",
        dataset_path="RAF-DB",
        template_path="template/emotion_classification.json",
    ):
        self.root_path = root_path
        self.data_path = dataset_path
        self.template_path = template_path
        self.emotion_labels = [
            "Surprise",
            "Fear",
            "Disgust",
            "Happiness",
            "Sadness",
            "Anger",
            "Neutral",
        ]

    def build(self):
        dataset = []
        # Load the template
        question_list = json.load(open(self.template_path, "r"))

        # Read the labels from train_label.mat
        label_file_path = os.path.join(
            self.root_path, self.data_path, "RAF_labels/train_label.mat"
        )
        mat_data = scipy.io.loadmat(label_file_path)
        labels = mat_data["train_label"]

        # Read the image names from train_name.txt
        name_file_path = os.path.join(
            self.root_path, self.data_path, "RAF_labels/train_name.txt"
        )
        with open(name_file_path, "r") as file:
            image_names = file.read().splitlines()

        # Process each image and label
        for i, (image_name, label) in enumerate(zip(image_names, labels)):
            image_path = os.path.join(
                self.root_path,
                self.data_path,
                f"RAF_images/crop_images",
                image_name,
            )

            # Randomly select a question from the template
            question = random.choice(question_list)

            # Create the entry for the dataset
            entry = {
                "id": image_name.split(".")[0],
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": question},
                    {
                        "from": "gpt",
                        "value": f"The emotion of this image is {self.emotion_labels[int(label) - 1].lower()}",
                    },
                ],
            }

            # Append to the dataset
            dataset.append(entry)

        return dataset
