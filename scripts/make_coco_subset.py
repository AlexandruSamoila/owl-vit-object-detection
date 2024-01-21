import json
import random
from collections import Counter, OrderedDict, defaultdict
from copy import copy

import yaml

# This is because the coco classes have weird numbering
convertor = {
    1: {"new_idx": 0, "name": "ceres"},
    2: {"new_idx": 1, "name": "Oddset"},

}

new_labelmap = {element["new_idx"]: element["name"] for element in convertor.values()}


def load_config():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)["data"]
        source_annotations_file = data["annotations_file"]
        num_train_samples = data["num_train_images"]
        num_test_samples = data["num_test_images"]

    return source_annotations_file, num_train_samples, num_test_samples


def shuffle_indices(subset_indices, num_train_samples, num_test_samples):
    random.shuffle(subset_indices)
    train_indices = subset_indices[:num_train_samples]
    test_indices = subset_indices[
        num_train_samples : num_train_samples + num_test_samples
    ]
    return train_indices, test_indices


if __name__ == "__main__":
    source_annotations_file, num_train_samples, num_test_samples = load_config()

    with open(source_annotations_file) as f:
        dat = json.load(f)
        images = dat["images"]
        annotations = dat["annotations"]

    _annotations = defaultdict(list)
    for annotation in annotations:
        _annotations[annotation["image_id"]].append(
            {
                "bbox": annotation["bbox"],
                "label": convertor[annotation["category_id"]]["new_idx"],
            }
        )
    annotations = _annotations

    subset_indices = [i["id"] for i in images]
    train_indices, test_indices = shuffle_indices(
        subset_indices, num_train_samples, num_test_samples
    )
    train_imagemap = {
        element["id"]: element["file_name"]
        for element in images
        if element["id"] in train_indices
    }

    test_imagemap = {
        element["id"]: element["file_name"]
        for element in images
        if element["id"] in test_indices
    }

    print("Searching for a valid subset...")
    train = {}
    test = {}
    while True:
        classes = []
        for id, fpath in train_imagemap.items():
            train[fpath] = annotations[id]
            classes.extend([new_labelmap[el["label"]] for el in annotations[id]])

        for id, fpath in test_imagemap.items():
            test[fpath] = annotations[id]
            classes.extend([new_labelmap[el["label"]] for el in annotations[id]])
        classcounts = OrderedDict(Counter(classes).most_common())
        print(json.dumps(classcounts, indent=2))
        accept = input("accept? (y/n) >")

        if accept == "y":
            break
        else:
            print("Searching for a valid subset (this might take a few seconds)...")
            subset_indices = [i["id"] for i in images]
            train_indices, test_indices = shuffle_indices(
                subset_indices, num_train_samples, num_test_samples
            )
            train_imagemap = {
                element["id"]: element["file_name"]
                for element in images
                if element["id"] in train_indices
            }

            test_imagemap = {
                element["id"]: element["file_name"]
                for element in images
                if element["id"] in test_indices
            }

    with open("data/train.json", "w") as f:
        json.dump(train, f)

    with open("data/test.json", "w") as f:
        json.dump(test, f)

    with open("data/counts.json", "w") as f:
        json.dump(classcounts, f)

    with open("data/labelmap.json", "w") as f:
        json.dump(new_labelmap, f)
