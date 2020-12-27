import numpy as np
from collections import deque

class SegmentationNode(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.nodes = []
    
def check_node_homogeneous(node, image, min_region_square, treshold):
    node_size = node.height * node.width
    if node_size < 16:
        return True
    
    x_end = node.x + node.width
    y_end = node.y + node.height

    r = image[node.y:y_end, node.x:x_end, :]

    gray = 0.2126 * r[:, :, 0] + 0.7152 * r[:, :, 1] + \
                    0.0722 * r[:, :, 2]
    
    region_mean = np.mean(gray)

    v = np.sum((gray - region_mean) ** 2) / (node_size - 1)

    return v < treshold


def split(node, image, treshold):

    split_candidates = deque()
    split_candidates.append(node)

    while split_candidates:
        to_split = split_candidates.popleft()
        if not check_node_homogeneous(to_split, image, 16, treshold):
            x = to_split.x
            y = to_split.y
            width = to_split.width
            w2 = width//2
            height = to_split.height
            h2 = height//2

            top_left = SegmentationNode(x, y, w2, h2)
            top_right = SegmentationNode(x + w2, y, width - w2, h2)
            down_left = SegmentationNode(x, y + h2, w2, height - h2)
            down_right = SegmentationNode(x + w2, y + h2, width - w2, height - h2)
            nodes = [top_left, top_right, down_left, down_right]
            to_split.nodes = nodes
            split_candidates.extend(nodes)
    
def merge(node, image):
    if node.nodes:
        for n in node.nodes:
            merge(n, image)
    else:
        node_size = node.height * node.width
        x_end = node.x + node.width
        y_end = node.y + node.height

        region = image[node.y:y_end, node.x:x_end, :]
        r = region[:, :, 0]
        g = region[:, :, 1]
        b = region[:, :, 2]
        
        image[node.y:y_end, node.x:x_end, 0] = np.mean(r)
        image[node.y:y_end, node.x:x_end, 1] = np.mean(g)
        image[node.y:y_end, node.x:x_end, 2] = np.mean(b)


def segment_image(image, treshold):
    height, width = image.shape[:2]
    root = SegmentationNode(0, 0, width, height)
    split(root, image, treshold)
    merge(root, image)
    return image