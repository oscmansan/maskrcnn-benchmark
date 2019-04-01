import argparse
import json
import os

from PIL import Image
from functional import seq


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        'input', help="Udacity dataset path", type=str)
    parser.add_argument(
        'output', help="output dir for json files", type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    sets = ['test', 'train', 'valid']

    for set in sets:
        root = os.path.join(args.input, set)

        images = seq(os.listdir(root)).filter(lambda f: f.endswith('.jpg')).to_list()
        annotations = seq(os.listdir(root)).filter(lambda f: f.endswith('.jpg')).to_list()

        info_json = {
            "description": 'Udacity',
        }
        images_json = []
        annotations_json = []
        categories_json = [
            {'id': 0, 'name': 'Car'},
            {'id': 1, 'name': 'Pedestrian'},
            {'id': 2, 'name': 'Truck'},
        ]

        annotation_id = 0

        for id, image_path, annotation_path in enumerate(zip(images, annotations)):
            width, height = Image.open(image_path).size

            with open(annotation_path, 'r') as f:
                bboxes = (seq(f.readlines())
                          .map(lambda l: l.split(' '))
                          .map(lambda p: (int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
                          .to_list())

            images_json.append({
                'id': id,
                'width': width,
                'height': height,
                'file_name': image_path
            })

            for bbox in bboxes:
                annotations_json.append({
                    'id': annotation_id,
                    'category_id': bbox[0],
                    'image_id': id,
                    'bbox': [bbox[1] - bbox[3] // 2, bbox[2] - bbox[4] // 2, bbox[3], bbox[4]],
                    'area': bbox[3] * bbox[4],
                    'iscrowd': 0
                })
                annotation_id += 1

        json_dict = {
            'info': info_json,
            'images': images_json,
            'annotations': annotations_json,
            'categories': categories_json,
            'licenses': []
        }

        print(json.dumps(json_dict))


if __name__ == '__main__':
    main()
