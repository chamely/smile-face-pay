import cv2
import numpy as np
from skimage import transform as trans

REFERENCE = np.array([
    [38.29, 51.69], [73.53, 51.50],
    [56.02, 71.73], [41.55, 92.36], [70.72, 92.20]
], dtype=np.float32)

def align_face(image_bgr, kps, size=112):
    tform = trans.SimilarityTransform()
    tform.estimate(kps, REFERENCE)
    M = tform.params[0:2, :]
    return cv2.warpAffine(image_bgr, M, (size, size), borderValue=0.0)