#!/usr/bin/env python

"""
Detect thresholded saliency maps for all image in specified folder.
"""

__author__    = 'Igor Ryabtsov aka Tinnulion'
__copyright__ = 'Copyright (c) 2014'
__license__   = 'Apache 2.0'
__version__   = '1.0'

import os
from PIL import Image
import SightSpotDetector

# INPUT_FOLDER = '/Users/KimTS/Workspace/MILA/kaggle_whale/data/imgs/'
# OUTPUT_FOLDER = './set/whales/'
INPUT_FOLDER = '/data/lisatmp4/whales/data/imgs/'
OUTPUT_FOLDER = '/data/lisatmp4/taesup/whales'

for root, dirs, files in os.walk(INPUT_FOLDER, topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if path.find('.jpg')==-1:
            continue
        print 'Processing:', path
        detector = SightSpotDetector.SightSpotDetector(path)

        foreground = detector.get_foreground(source='precise')
        foreground.save(OUTPUT_FOLDER + '/cropped_' + name)




