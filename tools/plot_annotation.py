import argparse
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches

parser = argparse.ArgumentParser(description='Convert dataset')
parser.add_argument('image')
parser.add_argument('annotation')

args = parser.parse_args()

image = Image.open(args.image)

bboxes = []
with open(args.annotation, 'r') as f:
    for line in f.readlines():
        s = line.split(' ')
        bboxes.append([int(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4])])

plt.figure()
plt.imshow(image)
for bbox in bboxes:
    rect = patches.Rectangle(((bbox[1] - bbox[3] / 2) * image.width, (bbox[2] - bbox[4] / 2) * image.height),
                             bbox[3] * image.width, bbox[4] * image.height,
                             linewidth=2, edgecolor='blue', facecolor='none')
    plt.gca().add_patch(rect)

plt.show()
