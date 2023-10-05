from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import numpy as np

torch.set_grad_enabled(False)
# root to test images
images = Path('/mnt/home_6T/public/weien/area1/')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'

# load LightGlue feature extractor & matcher
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features='superpoint').eval().to(device)
# load 2 test images
image0 = load_image(images / '1.12.png')
image1 = load_image(images / '1.15.png')
# extract descriptors from 2 test images
feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))
# match 2 descriptors from 2 test images
matches01 = matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']


# shape [n, 2], n = number of matching points
print(matches.shape)

# matching visualization
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
viz2d.save_plot('/mnt/home_6T/public/weien/1215.png')

kpc0, kpc1 = viz2d.cm_prune(matches01['prune0']), viz2d.cm_prune(matches01['prune1'])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
viz2d.save_plot('/mnt/home_6T/public/weien/out_img.png')