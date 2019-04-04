import os
import argparse
import json
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
import visdom


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir')
    parser.add_argument('ann_file')
    return parser.parse_args()


def main():
    args = parse_args()

    vis = visdom.Visdom()

    with open(args.ann_file, 'r') as f:
        data = json.load(f)

    img_to_ann = defaultdict(list)
    for ann in data['annotations']:
        img_to_ann[ann['image_id']].append(ann)

    images = data['images']
    images = np.random.choice(images, 10)
    for img in images:
        image = Image.open(os.path.join(args.img_dir, img['file_name']))
        draw = ImageDraw.Draw(image)
        for ann in img_to_ann[img['id']]:
            bbox = ann['bbox']
            draw.rectangle([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], outline=(0, 255, 0))
        vis.image(np.array(image).transpose((2, 0, 1)))


if __name__ == '__main__':
    main()