import argparse
import json
import os

from PIL import Image
from functional import seq


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        'input', help="TT100K dataset path", type=str)
    parser.add_argument(
        'output', help="output dir for json files", type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    sets = ['test', 'train', 'valid']

    for image_set in sets:
        print('Converting set: {}...'.format(image_set))
        root = os.path.join(args.input, image_set)

        images = (seq(os.listdir(root))
                  .filter(lambda name: name.endswith('.jpg'))
                  .map(lambda name: os.path.join(root, name))
                  .to_list())
        annotations = (seq(os.listdir(root))
                       .filter(lambda name: name.endswith('.txt'))
                       .map(lambda name: os.path.join(root, name))
                       .to_list())

        info_json = {
            "description": 'TT100K',
        }
        images_json = []
        annotations_json = []
        categories_json = [
            {'id': 0, 'name': 'i2'},
            {'id': 1, 'name': 'i4'},
            {'id': 2, 'name': 'i5'},
            {'id': 3, 'name': 'il100'},
            {'id': 4, 'name': 'il60'},
            {'id': 5, 'name': 'il80'},
            {'id': 6, 'name': 'io'},
            {'id': 7, 'name': 'ip'},
            {'id': 8, 'name': 'p10'},
            {'id': 9, 'name': 'p11'},
            {'id': 10, 'name': 'p12'},
            {'id': 11, 'name': 'p19'},
            {'id': 12, 'name': 'p23'},
            {'id': 13, 'name': 'p26'},
            {'id': 14, 'name': 'p27'},
            {'id': 15, 'name': 'p3'},
            {'id': 16, 'name': 'p5'},
            {'id': 17, 'name': 'p6'},
            {'id': 18, 'name': 'pg'},
            {'id': 19, 'name': 'ph4'},
            {'id': 20, 'name': 'ph4.5'},
            {'id': 21, 'name': 'ph5'},
            {'id': 22, 'name': 'pl100'},
            {'id': 23, 'name': 'pl120'},
            {'id': 24, 'name': 'pl20'},
            {'id': 25, 'name': 'pl30'},
            {'id': 26, 'name': 'pl40'},
            {'id': 27, 'name': 'pl5'},
            {'id': 28, 'name': 'pl50'},
            {'id': 29, 'name': 'pl60'},
            {'id': 30, 'name': 'pl70'},
            {'id': 31, 'name': 'pl80'},
            {'id': 32, 'name': 'pm20'},
            {'id': 33, 'name': 'pm30'},
            {'id': 34, 'name': 'pm55'},
            {'id': 35, 'name': 'pn'},
            {'id': 36, 'name': 'pne'},
            {'id': 37, 'name': 'po'},
            {'id': 38, 'name': 'pr40'},
            {'id': 39, 'name': 'w13'},
            {'id': 40, 'name': 'w32'},
            {'id': 41, 'name': 'w55'},
            {'id': 42, 'name': 'w57'},
            {'id': 43, 'name': 'w59'},
            {'id': 44, 'name': 'wo'},
        ]

        annotation_id = 0

        for id, (image_path, annotation_path) in enumerate(zip(images, annotations)):
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
                'file_name': os.path.basename(image_path)
            })

            for bbox in bboxes:
                annotations_json.append({
                    'id': annotation_id,
                    'category_id': bbox[0],
                    'image_id': id,
                    'bbox': [(bbox[1] - bbox[3] // 2) * width, (bbox[2] - bbox[4] // 2) * height, bbox[3] * width,
                             bbox[4] * height],
                    'area': bbox[3] * bbox[4],
                    'iscrowd': 0,
                    'segmentation': []
                })
                annotation_id += 1

        json_dict = {
            'info': info_json,
            'images': images_json,
            'annotations': annotations_json,
            'categories': categories_json,
            'licenses': []
        }

        with open(os.path.join(args.output, 'instances_{}.json'.format(image_set)), 'w') as f:
            json.dump(json_dict, f)


if __name__ == '__main__':
    main()
