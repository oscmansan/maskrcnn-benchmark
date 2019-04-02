import os
import argparse
import json

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Udacity dataset to COCO format.')
    parser.add_argument('img_dir', type=str, help="Udacity dataset path.")
    parser.add_argument('ann_dir', type=str, help="Output dir for annotation files.")
    return parser.parse_args()


def main():
    args = parse_args()

    sets = ['test', 'train', 'valid']

    for image_set in sets:
        print('Converting set: {}...'.format(image_set))
        root = os.path.join(args.img_dir, image_set)

        images = sorted([os.path.join(root, name) for name in os.listdir(root) if name.endswith('.jpg')])
        annotations = sorted([os.path.join(root, name) for name in os.listdir(root) if name.endswith('.txt')])

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

        for id, (image_path, annotation_path) in enumerate(zip(images, annotations)):
            width, height = Image.open(image_path).size

            with open(annotation_path, 'r') as f:
                bboxes = []
                for l in f.readlines():
                    p = l.split(' ')
                    bboxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))

            images_json.append({
                'id': id,
                'width': width,
                'height': height,
                'file_name': os.path.basename(image_path)
            })

            for bbox in bboxes:
                category_id = bbox[0]
                x = (bbox[1] - bbox[3] / 2) * width
                y = (bbox[2] - bbox[4] / 2) * height
                w = bbox[3] * width
                h = bbox[4] * height

                annotations_json.append({
                    'id': annotation_id,
                    'category_id': category_id,
                    'image_id': id,
                    'bbox': [x, y, w, h],
                    'area': w * h,
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

        with open(os.path.join(args.ann_dir, 'instances_{}.json'.format(image_set)), 'w') as f:
            json.dump(json_dict, f)


if __name__ == '__main__':
    main()
