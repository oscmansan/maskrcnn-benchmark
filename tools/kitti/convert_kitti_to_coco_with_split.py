import argparse
import json
import os
import random

from PIL import Image

SPLIT_RATIO = 0.8


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Udacity dataset to COCO format.')
    parser.add_argument('img_dir', type=str, help="Udacity dataset path.")
    parser.add_argument('ann_dir', type=str, help="Output dir for annotation files.")
    return parser.parse_args()


def main():
    args = parse_args()

    root = os.path.join(args.img_dir, 'training')
    images_root = os.path.join(root, 'image_2')
    labels_root = os.path.join(root, 'label_2')
    images = sorted([os.path.join(images_root, name) for name in os.listdir(images_root) if name.endswith('.png')])
    annotations = sorted([os.path.join(labels_root, name) for name in os.listdir(labels_root) if name.endswith('.txt')])

    info_json = {
        "description": 'Kitti',
    }
    images_json_train = []
    images_json_test = []
    annotations_json_train = []
    annotations_json_test = []

    categories_map = {}
    categories_json = []

    image_id_train = 0
    image_id_test = 0
    annotation_id_train = 0
    annotation_id_test = 0

    for image_path, annotation_path in zip(images, annotations):
        width, height = Image.open(image_path).size

        is_train = random.random() < SPLIT_RATIO

        with open(annotation_path, 'r') as f:
            bboxes = []
            for l in f.readlines():
                p = l.split(' ')
                bboxes.append((p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4])))

        image = {
            'id': image_id_train if is_train else image_id_test,
            'width': width,
            'height': height,
            'file_name': os.path.basename(image_path)
        }

        annotation = []

        for bbox in bboxes:
            if bbox[0] not in categories_map.keys():
                categories_json.append({'name': bbox[0], 'id': len(categories_map.keys())})
                categories_map[bbox[0]] = len(categories_map.keys())

            category_id = categories_map[bbox[0]]
            x = (bbox[1] - bbox[3] / 2) * width
            y = (bbox[2] - bbox[4] / 2) * height
            w = bbox[3] * width
            h = bbox[4] * height

            annotation.append({
                'id': annotation_id_train if is_train else annotation_id_test,
                'category_id': category_id,
                'image_id': image_id_train if is_train else image_id_test,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'segmentation': []
            })
            if is_train:
                annotation_id_train += 1
            else:
                annotation_id_test += 1

        if is_train:
            images_json_train.append(image)
            annotations_json_train += annotation
            image_id_train += 1
        else:
            images_json_test.append(image)
            annotations_json_test += annotation
            image_id_test += 1

    json_dict_train = {
        'info': info_json,
        'images': images_json_train,
        'annotations': annotations_json_train,
        'categories': categories_json,
        'licenses': []
    }

    json_dict_test = {
        'info': info_json,
        'images': images_json_test,
        'annotations': annotations_json_test,
        'categories': categories_json,
        'licenses': []
    }

    with open(os.path.join(args.ann_dir, 'instances_train.json'), 'w') as f:
        json.dump(json_dict_train, f)

    with open(os.path.join(args.ann_dir, 'instances_test.json'), 'w') as f:
        json.dump(json_dict_test, f)


if __name__ == '__main__':
    main()
