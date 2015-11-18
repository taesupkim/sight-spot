#!/usr/bin/env python

"""
SightSpotDetector is fully-functional visual attention framework.

It allows the following:
1) Calculation of saliency map for given image (grayscale or heatmap-colored).
2) Obtain pixel-precise saliency map of given image.
3) Extract main objects using specific algorithm.

Framework uses lazy calculation so it`s doing less worthless work.
"""

__author__    = 'Igor Ryabtsov aka Tinnulion'
__copyright__ = "Copyright (c) 2014"
__license__   = "Apache 2.0"
__version__   = "1.0"

import cv2
import math
import numpy
import scipy.ndimage
from PIL import Image
import SightSpotUtil

class SightSpotDetector():

    # Global constants.
    _SMALL_BLUR_RATIO = 0.1
    _LARGE_BLUR_RATIO = 0.15
    _CELL_SIZE_RATIO = 1.0
    _SLIC_ALPHA = 0.25
    _ITERATION_NUMBER = 4

    def __init__(self, image, grain=0.1):
        """
        Public constructor of SightSpotDetector.

        Parameters
        ----------
        image : PIL.Image, numpy.ndarray or string
            Should be either 3 channel RGB image with 8bit channels (PIL.Image or numpy array) or filename.
        grain : floating-point value
            Value should be between 0.0 and 1.0.
        """
        rgb_image = image
        if isinstance(rgb_image, basestring):
            rgb_image = Image.open(rgb_image)
        if isinstance(rgb_image, Image.Image):
            rgb_image = numpy.asarray(rgb_image, dtype='float32')
        if not isinstance(rgb_image, numpy.ndarray):
            raise Exception('Argument "image" should be either PIL.Image, numpy.ndarray or string filename!')

        rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.2, fy=0.2)
        # print rgb_image.shape
        org_image = cv2.copyMakeBorder(rgb_image, 100, 100, 100, 100, cv2.BORDER_REFLECT101)
        rgb_image = cv2.blur(org_image,(32,32))
        # print rgb_image.shape
        self._org_image = org_image
        self._rgb_image = rgb_image
        self._orgb_image = None
        self._saliency_map = None
        self._fusion_map = None
        self._raw_heatmap = None
        self._precise_heatmap = None

        (width, height) = self._rgb_image.shape[1], self._rgb_image.shape[0]
        area = float(width * height)
        dim = math.sqrt(area) * grain
        self._small_sigma = self._SMALL_BLUR_RATIO * dim
        self._large_sigma = self._LARGE_BLUR_RATIO * dim
        self._cell_size = self._CELL_SIZE_RATIO * dim
        if self._cell_size <= 4.0:
            self._cell_size = 4.0

    def _get_orgb_image(self):
        if self._orgb_image is None:
            self._orgb_image = SightSpotUtil.eval_orgb_image(self._rgb_image)
        return self._orgb_image

    def _get_saliency_map(self, type):
        orgb_image = self._get_orgb_image()
        if type == 'raw':
            if self._saliency_map is None:
                small_sigma = self._small_sigma
                large_sigma = self._large_sigma
                self._saliency_map = SightSpotUtil.eval_saliency_map(orgb_image, small_sigma, large_sigma, 'auto')
            return self._saliency_map
        if type == 'precise':
            if self._fusion_map is None:
                saliency_map = self._get_saliency_map(type='raw')
                cell_size = self._cell_size
                alpha = self._SLIC_ALPHA
                iterations = self._ITERATION_NUMBER
                segmentation_map = SightSpotUtil.eval_slic_map(orgb_image, cell_size, alpha, iterations)
                self._fusion_map = SightSpotUtil.combine_saliency_and_segmentation(saliency_map, segmentation_map)
            return self._fusion_map
        raise Exception('Unknown argument value type = "' + str(type) + '"')

    def get_saliency_map(self, type='precise'):
        """
        Returns saliency map of image passed to SightSpotDetector constructor.

        Parameters
        ----------
        type : 'raw' or 'precise'
            Specifies whenever segmentation will be used to improve saliency map (slower).

        Returns
        -------
        out : ndarray
            Saliency map.
        """
        return self._get_saliency_map(type).copy()

    def get_heatmap(self, type='raw'):
        """
        Returns heatmap calculated from saliency map of image passed to SightSpotDetector constructor.

        Parameters
        ----------
        type : 'raw' or 'precise'
            Specifies whenever segmentation will be used to improve saliency map (slower).

        Returns
        -------
        out : PIL.Image
            Heatmap.
        """
        if type == 'raw':
            if self._raw_heatmap == None:
                saliency_map = self._get_saliency_map(type)
                self._raw_heatmap = SightSpotUtil.eval_heatmap(saliency_map)
            return self._raw_heatmap.copy()
        if type == 'precise':
            if self._precise_heatmap == None:
                fusion_map = self._get_saliency_map(type)
                self._precise_heatmap = SightSpotUtil.eval_heatmap(fusion_map)
            return self._precise_heatmap.copy()
        raise Exception('Unknown argument value type = "' + str(type) + '"')

    def threshold(self, source, value='auto'):
        """
        Returns black-and-white image - thresholded version of saliency map.

        Parameters
        ----------
        source : 'raw' or 'precise'
            Which saliency map to use for thresholding.
        type : float or 'auto'
            Specifies thresholding value.

        Returns
        -------
        out : PIL.Image
            Binary image.
        """
        saliency_map = self._get_saliency_map(source)
        return Image.fromarray(255.0 * SightSpotUtil.threshold(saliency_map, value)).convert('RGB')

    def get_foreground(self, source, value='auto'):
        """
        Removes background pixels from input image and returns result.

        Parameters
        ----------
        source : 'raw' or 'precise'
            Which saliency map to use for thresholding.
        type : float or 'auto'
            Specifies thresholding value.

        Returns
        -------
        out : PIL.Image
            Image without background pixels.
        """
        saliency_map = self._get_saliency_map(source)
        return Image.fromarray(SightSpotUtil.remove_background(self._org_image, saliency_map, value))

    def cut_objects(self, source, value='auto'):
        """
        Extract connected components from thresholded saliency map and cuts images.

        Parameters
        ----------
        source : 'raw' or 'precise'
            Which saliency map to use for thresholding.
        type : float or 'auto'
            Specifies thresholding value.

        Returns
        -------
        out : list of PIL.Images
            Detected objects.
        """
        saliency_map = self._get_saliency_map(source)
        result = SightSpotUtil.detect_objects(self._rgb_image, saliency_map, value)
        windows = []
        for item in result:
            windows.append(Image.fromarray(item))
        return windows

if __name__ == "__main__":
    print __doc__


