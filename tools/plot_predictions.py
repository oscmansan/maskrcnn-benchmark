import os
import argparse
import json

import numpy as np
import cv2
import visdom

import torch
from torchvision import transforms as T

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import Checkpointer
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str)
    parser.add_argument('weights_file', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--confidence-thresh', type=float, default=0.5)
    parser.add_argument('--num-images', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    vis = visdom.Visdom()

    # Load configuration file.
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(['MODEL.MASK_ON', False])
    cfg.merge_from_list(['TEST.IMS_PER_BATCH', 1])

    # Build model.
    model = build_detection_model(cfg).cuda()
    model.eval()

    # Load weights from checkpoint.
    checkpointer = Checkpointer(model)
    checkpointer.load(args.weights_file)

    # Build pre-processing transform.
    transforms = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(cfg.INPUT.MIN_SIZE_TEST),
            T.ToTensor(),
            T.Lambda(lambda x: x * 255),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ]
    )

    # Load dataset.
    attrs = DatasetCatalog.get(args.dataset)['args']
    img_dir = attrs['root']
    ann_file = attrs['ann_file']
    with open(ann_file, 'r') as f:
        dataset = json.load(f)

    images = dataset['images']
    categories = [category['name'] for category in dataset['categories']]
    categories.insert(0, '__background')

    for img in np.random.choice(images, args.num_images):
        # Load image in OpenCV format.
        image_file = os.path.join(img_dir, img['file_name'])
        original_image = cv2.imread(image_file)
        print(image_file)

        # Apply pre-processing to image.
        image = transforms(original_image)
        image = image.to('cuda')

        # Compute predictions.
        with torch.no_grad():
            predictions = model(image)
        predictions = predictions[0]
        predictions = predictions.to('cpu')

        # Reshape prediction into the original image size.
        height, width = original_image.shape[:-1]
        predictions = predictions.resize((width, height))

        # Select top predictions.
        keep = torch.nonzero(predictions.get_field("scores") > args.confidence_thresh).squeeze(1)
        predictions = predictions[keep]

        scores = predictions.get_field('scores').tolist()
        labels = predictions.get_field('labels').tolist()
        labels = [categories[i] for i in labels if i < len(categories)]
        boxes = predictions.bbox

        # Compose result image.
        result = original_image.copy()
        template = '{}: {:.2f}'
        for box, score, label in zip(boxes, scores, labels):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            cv2.rectangle(result, tuple(top_left), tuple(bottom_right), (0, 255, 0), 1)
            s = template.format(label, score)
            cv2.putText(result, s, tuple(top_left), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        # Visualize image in Visdom.
        vis.image(result[:, :, ::-1].transpose((2, 0, 1)), opts=dict(title=image_file))


if __name__ == '__main__':
    main()
