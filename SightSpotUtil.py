#!/usr/bin/env python

"""
This file contains implementation of few auxiliary algorithms for SightSpotDetector.
"""

__author__    = 'Igor Ryabtsov aka Tinnulion'
__copyright__ = "Copyright (c) 2014"
__license__   = "Apache 2.0"
__version__   = "1.0"

import math
import sys
import numpy
import numpy.linalg
import scipy.ndimage
import scipy.fftpack
from PIL import Image

def _eval_new_orgb_angle(rt, mask, add, mul):
    rt[mask] = add + mul * rt[mask]

def eval_orgb_image(rgb_image):
    """
    Converts specified RGB image to array with oRGB components.

    Parameters
    ----------
    rgb_image : ndarray
        Should be 3 channel RGB image with 8bit per channels

    Returns
    -------
    out : ndarray
        oRGB representation of image.
    """
    assert(len(rgb_image.shape) == 3)
    assert(rgb_image.shape[-1] == 3)
    h, w = rgb_image.shape[0:2]
    orgb_array = numpy.reshape(numpy.array(rgb_image, dtype='float32') / 255.0, (h * w, 3))
    pre_transform = numpy.array([[0.2990, 0.5870, 0.1140], [0.8660, -0.8660, 0.0], [0.5000, 0.5000, -1.0000]])
    orgb_array = numpy.dot(pre_transform, orgb_array.transpose()).transpose()

    rg = orgb_array[:,1]
    yb = orgb_array[:,2]
    t = numpy.arctan2(rg, yb)
    mask_t_vs_0 = (t >= 0.0)
    mask_t_vs_pi_3 = (numpy.abs(t) >= math.pi / 3.0)
    rt = numpy.array(t)

    _eval_new_orgb_angle(rt, mask_t_vs_0 & mask_t_vs_pi_3, 0.25 * math.pi, 0.75)
    _eval_new_orgb_angle(rt, mask_t_vs_0 & ~mask_t_vs_pi_3, 0.0, 1.5)

    _eval_new_orgb_angle(rt, ~mask_t_vs_0 & mask_t_vs_pi_3, -0.25 * math.pi, 0.75)
    _eval_new_orgb_angle(rt, ~mask_t_vs_0 & ~mask_t_vs_pi_3, 0.0, 1.5)

    dt = rt - t
    cos_dt = numpy.cos(dt)
    sin_dt = numpy.sin(dt)
    r_yb = cos_dt * yb - sin_dt * rg
    r_rg = sin_dt * yb + cos_dt * rg

    orgb_array[:,1] = r_rg
    orgb_array[:,2] = r_yb
    orgb_image = orgb_array.reshape(h, w, 3)
    return orgb_image

################################################################################

def _ft_gaussian_blur(fft_image, sigma, pad_x, pad_y, w, h):
    convolved_image =  scipy.ndimage.fourier_gaussian(fft_image, [sigma, sigma, 0.0])
    blurred_image = scipy.fftpack.ifft2(convolved_image, axes=(0, 1)).real
    cut_image = blurred_image[pad_y:pad_y+h, pad_x:pad_x+w,:]
    return cut_image

def eval_saliency_map(orgb_image, small_sigma, large_sigma, params=(0.0, 0.0, 0.0)):
    """
    Calculates raw saliency map from given oRGB image.

    Parameters
    ----------
    orgb_image : ndarray
        Should be 3 channel oRGB image with float32 channels
    small_sigma : float
        Controls blur - it`s best to set ~ 1% of input image dimension.
    large_sigma : float
        Controls blur - it`s best to set ~ 20% of input image dimension.
    params : 3-tuple of floats or 'auto' (can be omited - no correction this way)
        Nonlinear correction is applied to saliency map to make it perceive better.

    Returns
    -------
    out : ndarray
        Raw saliency map of input image.
    """
    assert(len(orgb_image.shape) == 3)
    assert(orgb_image.shape[-1] == 3)
    assert(large_sigma > small_sigma)
    assert((type(params) == tuple and len(params) == 3) or params == 'auto')

    w = orgb_image.shape[1]
    h = orgb_image.shape[0]
    pad_x = w / 8
    pad_y = h / 8
    padded_image = numpy.zeros((h + 2 * pad_y, w + 2 * pad_x, 3), dtype='float32')
    padded_image[pad_y:pad_y+h, pad_x:pad_x+w,:] = orgb_image[:,:,:]
    fft_image = scipy.fftpack.fft2(padded_image, axes=(0, 1))

    small_blur_image_1 = _ft_gaussian_blur(fft_image, 1 * small_sigma, pad_x, pad_y, w, h)
    small_blur_image_2 = _ft_gaussian_blur(fft_image, 2 * small_sigma, pad_x, pad_y, w, h)
    small_blur_image_4 = _ft_gaussian_blur(fft_image, 4 * small_sigma, pad_x, pad_y, w, h)

    large_blur_image_1 = _ft_gaussian_blur(fft_image, 1 * large_sigma, pad_x, pad_y, w, h)
    large_blur_image_2 = _ft_gaussian_blur(fft_image, 2 * large_sigma, pad_x, pad_y, w, h)
    large_blur_image_4 = _ft_gaussian_blur(fft_image, 4 * large_sigma, pad_x, pad_y, w, h)

    difference_1 = small_blur_image_1 - large_blur_image_1
    difference_2 = small_blur_image_2 - large_blur_image_2
    difference_4 = small_blur_image_4 - large_blur_image_4

    saliency_map_1 = numpy.sqrt(numpy.sum(numpy.square(difference_1), axis=2))
    saliency_map_2 = numpy.sqrt(numpy.sum(numpy.square(difference_2), axis=2))
    saliency_map_4 = numpy.sqrt(numpy.sum(numpy.square(difference_4), axis=2))

    saliency_map = (saliency_map_1 + saliency_map_2 + saliency_map_4) / 9.0
    if params == 'auto':
        a = numpy.mean(saliency_map) / 2.0
        b = numpy.std(saliency_map)
        c = 1.0
    else:
        a = float(params[0])
        b = float(params[1])
        c = float(params[2])
    saliency_map_log = 1.0 / (1.0 + numpy.exp(-(saliency_map - b) / a))
    saliency_map_adj = (1.0 - c) * saliency_map + c * saliency_map_log
    return saliency_map_adj

################################################################################

def _get_lowest_grad_pos(orgb_image, x, y):
    nx, ny = int(x + 0.5), int(y + 0.5)
    low_x, high_x = nx - 2, nx+ 3
    assert (low_x >= 0)
    assert (high_x < orgb_image.shape[1])
    low_y, high_y = ny - 2, ny + 3
    assert (low_y >= 0)
    assert (high_y < orgb_image.shape[0])
    neighbor = orgb_image[low_y:high_y, low_x:high_x]
    opt_x, opt_y = nx, ny
    min_grad = sys.float_info.max
    for dy in xrange(1, 4):
        for dx in xrange(1, 4):
            diff_x = numpy.mean(neighbor[dy-1:dy+2, dx-1] - neighbor[dy-1:dy+2, dx+1], axis=0)
            diff_y = numpy.mean(neighbor[dy-1, dx-1:dx+2] - neighbor[dy+1, dx-1:dx+2], axis=0)
            grad = numpy.linalg.norm(diff_x, 2) + numpy.linalg.norm(diff_y, 2)
            if grad < min_grad:
                opt_x, opt_y = nx + dx - 2, ny + dy - 2
                min_grad = grad
    return opt_x, opt_y

def _init_clusters_centers(orgb_image, cell_size):
    width = orgb_image.shape[1]
    height = orgb_image.shape[0]
    cell_number_x = int(float(width) / cell_size - 0.5)
    cell_number_y = int(float(height) / cell_size - 0.5)
    pad_x = 0.5 * (width - cell_number_x * cell_size)
    pad_y = 0.5 * (height - cell_number_y * cell_size)
    assert(pad_x >= 1.0)
    assert(pad_y >= 1.0)
    cluster_centers = []
    for ny in xrange(cell_number_y + 1):
        y = ny * cell_size + pad_y
        for nx in xrange(cell_number_x + 1):
            x = nx * cell_size + pad_x
            lx, ly = _get_lowest_grad_pos(orgb_image, x, y)
            cluster_centers.append((lx, ly, orgb_image[ly, lx]))
    return cluster_centers

def _do_slic_iteration(orgb_image, cell_size, labels, distances, cluster_centers, alpha):
    width = orgb_image.shape[1]
    height = orgb_image.shape[0]
    coordinates = numpy.mgrid[0:height, 0:width].swapaxes(0, 2).swapaxes(0, 1)

    for i in xrange(len(cluster_centers)):
        center_x, center_y = cluster_centers[i][0], cluster_centers[i][1]
        low_x, high_x = int(center_x - cell_size), int(center_x + cell_size + 1)
        low_y, high_y = int(center_y - cell_size), int(center_y + cell_size + 1)
        low_x, high_x = max(0, low_x), min(width, high_x)
        low_y, high_y = max(0, low_y), min(height, high_y)
        window = orgb_image[low_y:high_y, low_x:high_x]
        window_distances = distances[low_y:high_y, low_x:high_x]

        color_diff = window - cluster_centers[i][2]
        color_dist = numpy.sqrt(numpy.sum(numpy.square(color_diff), axis=2)) / 3.0
        mesh_x = numpy.square((numpy.arange(low_x, high_x) - center_x) / cell_size)
        mesh_y = numpy.square((numpy.arange(low_y, high_y) - center_y) / cell_size)
        mesh_xx, mesh_yy = numpy.meshgrid(mesh_x, mesh_y)
        coordinate_dist = numpy.sqrt(mesh_xx + mesh_yy)
        total_dist = (1.0 - alpha) * color_dist + alpha * coordinate_dist

        threshold_idx = (total_dist < window_distances)
        window_distances[threshold_idx] = total_dist[threshold_idx]

        labels[low_y:high_y, low_x:high_x][threshold_idx] = i
        distances[low_y:high_y, low_x:high_x] = window_distances

    for i in xrange(len(cluster_centers)):
        current_cluster_idx = (labels == i)
        cluster_size = numpy.sum(current_cluster_idx)
        if cluster_size != 0:
            new_coordinates = numpy.sum(coordinates[current_cluster_idx], axis=0)
            new_x = new_coordinates[1] / cluster_size
            new_y = new_coordinates[0] / cluster_size
            new_color = numpy.sum(orgb_image[current_cluster_idx], axis=0) / cluster_size
            cluster_centers[i] = (new_x, new_y, new_color)

def _label_unassigned_pixels(labels):
    neg_idx = (labels < 0)
    neg_map = neg_idx.astype('int32')
    labeled_array, num_features = scipy.ndimage.label(neg_map, structure=[[1,1,1],[1,1,1],[1,1,1]])
    labeled_array[neg_idx] += numpy.max(labels)
    labeled_array += labels
    return labeled_array

def _get_adj_by_direction(labels, dx, dy, joint_clusters):
    assert(dx in {-1, 0, 1} and dy in {-1, 0, 1} and ((dx != 0) or (dy != 0)))
    width = labels.shape[1]
    height = labels.shape[0]
    start_x, stop_x = 0, width - abs(dx)
    start_y, stop_y = 0, height - abs(dy)
    if dx < 0:
        start_x -= dx
        stop_x -= dx
    if dy < 0:
        start_y -= dy
        stop_y -= dy
    direction_first = labels[start_y:stop_y, start_x:stop_x]
    direction_second = labels[start_y+dy:stop_y+dy, start_x+dx:stop_x+dx]
    assert(direction_first.shape == direction_second.shape)

    border_idx = (direction_first != direction_second)
    neighbor_one = list(numpy.extract(border_idx, direction_first).flatten())
    neighbor_two = list(numpy.extract(border_idx, direction_second).flatten())
    neighbors_zipped = zip(neighbor_one, neighbor_two)
    joint_clusters.update(neighbors_zipped)

def _get_adj_matrix(labels, number_of_components):
    joint_clusters = set()
    _get_adj_by_direction(labels, 1, 0, joint_clusters)
    _get_adj_by_direction(labels, 0, 1, joint_clusters)
    _get_adj_by_direction(labels, 1, 1, joint_clusters)
    _get_adj_by_direction(labels, -1, 1, joint_clusters)

    adj_matrix = dict()
    for i in xrange(number_of_components):
        adj_matrix[i] = set()
    for item in joint_clusters:
        adj_matrix[item[0]].add(item[1])
        adj_matrix[item[1]].add(item[0])
    return adj_matrix

def _get_component_masks(labels, number_of_clusters):
    labels_masks = []
    for i in xrange(number_of_clusters):
        labels_masks.append(labels == i)
    return labels_masks

def _find_broken(labels, number_of_clusters, masks, cluster_centers, cell_size):
    min_area = int(0.1 * cell_size * cell_size + 0.5)
    is_broken = numpy.array([True] * number_of_clusters, dtype='bool')
    for center in cluster_centers:
        x, y = center[0], center[1]
        is_broken[labels[y, x]] = False
    for i in xrange(number_of_clusters):
        component_idx = masks[i]
        if numpy.sum(component_idx) < min_area:
            is_broken[i] = True
    return list(numpy.arange(0, number_of_clusters)[is_broken])

def _get_cluster_colors(orgb_image, number_of_components, masks):
    component_colors = []
    for i in xrange(number_of_components):
        component_idx = masks[i]
        cluster_size = numpy.sum(component_idx)
        avg_color = numpy.sum(orgb_image[component_idx], axis=0) / cluster_size
        component_colors.append(avg_color)
    return component_colors

def _enforce_connectivity(orgb_image, labels, cluster_centers, cell_size):
    labels = _label_unassigned_pixels(labels)
    number_of_clusters = numpy.max(labels) + 1
    masks = _get_component_masks(labels, number_of_clusters)
    broken_clusters = _find_broken(labels, number_of_clusters, masks, cluster_centers, cell_size)
    if len(broken_clusters) > 0:
        adj_matrix = _get_adj_matrix(labels, number_of_clusters)
        cluster_colors = _get_cluster_colors(orgb_image, number_of_clusters, masks)
        for i in broken_clusters:
            current_color = cluster_colors[i]
            cluster_idx = masks[i]
            adj_clusters = adj_matrix[i]
            new_label = i
            min_distance = sys.float_info.max
            for j in adj_clusters:
                cluster_color = cluster_colors[j]
                distance = numpy.sqrt(numpy.sum(numpy.square(cluster_color - current_color))) / 3.0
                if distance < min_distance:
                    min_distance = distance
                    new_label = j
            labels[cluster_idx] = new_label
    unique_labels = numpy.unique(labels)
    mapping = numpy.zeros(number_of_clusters, dtype='int32')
    for i in xrange(len(unique_labels)):
        mapping[unique_labels[i]] = i
    labels = mapping[labels]
    return labels

def eval_slic_map(orgb_image, cell_size, alpha, iteration_number=16):
    """
    Calculates segmentation map of given oRGB image by SLIC algorithm.

    Parameters
    ----------
    orgb_image : ndarray
        Should be 3 channel oRGB image with float32 channels
    cell_size : int
        Controls number of clusters.
    alpha : float
        Define what is more important - cluster compactness or color similarity.
    iteration_number : int
        Number of iteration. Default value is 16.

    Returns
    -------
    out : ndarray
        Segmentation labels map for each pixel.
    """
    assert(cell_size >= 4.0)
    cluster_centers = _init_clusters_centers(orgb_image, cell_size)
    for k in xrange(iteration_number):
        labels = -1 * numpy.ones(orgb_image.shape[:2], dtype='int32')
        distances = sys.float_info.max * numpy.ones(orgb_image.shape[:2], dtype='float32')
        _do_slic_iteration(orgb_image, cell_size, labels, distances, cluster_centers, alpha)
    labels = _enforce_connectivity(orgb_image, labels, cluster_centers, cell_size)
    return labels

################################################################################

def _get_heatmap_palette():
    palette = []
    s = 1.0
    v = 255.0
    for idx in xrange(256):
        h = 240.0 * (1.0 - idx / 255.0)
        hi = math.floor(h / 60.0) % 6
        f =  (h / 60.0) - math.floor(h / 60.0)
        p = v * (1.0 - s)
        q = v * (1.0 - (f * s))
        t = v * (1.0 - ((1.0 - f) * s))
        aux_dict = {
            0: (v, t, p),
            1: (q, v, p),
            2: (p, v, t),
            3: (p, q, v),
            4: (t, p, v),
            5: (v, p, q)}
        r, g, b = aux_dict[hi]
        r = int(r + 0.5)
        g = int(g + 0.5)
        b = int(b + 0.5)
        palette.append((r, g, b))
    return palette

def eval_heatmap(saliency_map):
    """
    Calculates heatmap map from given saliency map.

    Parameters
    ----------
    saliency_map : ndarray
        Should be 2D-array with [0, 1] items.

    Returns
    -------
    out : PIL.Image
        Heatmap image.
    """
    assert(len(saliency_map.shape) == 2)
    palette = numpy.array(_get_heatmap_palette(), dtype='uint8')
    indices = numpy.array(255.0 * saliency_map + 0.5, dtype='uint8')
    heatmap = palette[indices]
    return Image.fromarray(heatmap)

def combine_saliency_and_segmentation(saliency_map, segmentation_map):
    """
    Combines saliency map and segmentation map producing pixel-precise segmentation.

    Parameters
    ----------
    saliency_map : ndarray
        Should be 2D-array with [0, 1] items.
    segmentation_map : ndarray
        Should be 2D-array with nonnegative integer items.

    Returns
    -------
    out : ndarray
        New saliency map with pixel precision.
    """
    palette = scipy.ndimage.mean(saliency_map, segmentation_map, numpy.arange(0, numpy.max(segmentation_map) + 1))
    palette = numpy.array(palette)
    return palette[segmentation_map]

################################################################################

def _estimate_threshold(saliency_map):
    return 1.5 * numpy.mean(saliency_map)

def _get_salient_mask(saliency_map, value='auto'):
    if value == 'auto':
        value = _estimate_threshold(saliency_map)
    assert(0.0 < value)
    binary_img = (saliency_map >= value)
    while binary_img.nonzero()[0].size==0:
        value *= 0.5
        binary_img = (saliency_map >= value)
    return binary_img

def threshold(saliency_map, value='auto'):
    """
    Thresholds (binarizes) saliency map.

    Parameters
    ----------
    saliency_map : ndarray
        Should be 2D-array with [0, 1] items.
    value : float or 'auto'
        Scalar should be in [0, 1].

    Returns
    -------
    out : ndarray
        List of thresholded maps.
    """
    idx = _get_salient_mask(saliency_map, value)
    threshold_map = idx.astype('int32')
    return threshold_map

def remove_background(rgb_image, saliency_map, value='auto'):
    """
    Substitute background pixels by black color.

    Parameters
    ----------
    rgb_image : ndarray
        Should be 3 channel RGB image with 8bit per channels
    saliency_map : ndarray
        Should be 2D-array with [0, 1] items.
    value : float or 'auto'
        Scalar should be in [0, 1].

    Returns
    -------
    out : ndarray
        List of thresholded maps.
    """
    idx = _get_salient_mask(saliency_map, value)

    result = rgb_image.copy()
    result = result[100:-100, 100:-100,:]
    idx=idx[100:-100, 100:-100]

    background_size = 0
    foreground_size = 0
    clusters_idx, num_clusters = scipy.ndimage.label(idx)
    for c in xrange(num_clusters):
        cluster_pos = numpy.argwhere(clusters_idx==c)
        cluster_size = cluster_pos.shape[0]
        if background_size<cluster_size:
            background_size=cluster_size
        elif foreground_size<cluster_size:
            foreground_size=cluster_size


    for c in xrange(num_clusters):
        cluster_pos = numpy.argwhere(clusters_idx==c)
        cluster_size = cluster_pos.shape[0]
        if cluster_size<foreground_size*0.5:
            idx[clusters_idx==c]=False

    idx=idx.nonzero()
    row_min =  min(idx[0])-50
    if row_min<0:
        row_min=0
    row_max =  max(idx[0])+50
    if row_max>=result.shape[0]:
        row_max = result.shape[0]-1
    col_min =  min(idx[1])-50
    if col_min<0:
        col_min=0
    col_max =  max(idx[1])+50
    if col_max>=result.shape[1]:
        col_max = result.shape[1]-1


    result = result[row_min:row_max, col_min:col_max,:]

    width, height = result.shape[1], result.shape[0]

    import cv2
    if width>height:
        dst_height = 256.
        dst_width  = (dst_height/height)*width
        result = cv2.resize(result, (int(dst_width),int(dst_height)))
    else:
        dst_width = 256.
        dst_height  = (dst_width/width)*height
        result = cv2.resize(result, (int(dst_width),int(dst_height)))

    return result.astype('uint8')

def detect_bounds(saliency_map, value='auto', type='rect'):
    """
    Detects bounds of salient objects (rect, square of circle).

    Parameters
    ----------
    saliency_map : ndarray
        Should be 2D-array with [0, 1] items.
    value : float or 'auto'
        Scalar should be in [0, 1].
    type : string
        'rect' (default) or 'square'

    Returns
    -------
    out : ndarray
        List of thresholded maps.
    """
    idx = _get_salient_mask(saliency_map, value)
    threshold_map = idx.astype('int32')
    components, number_of_components = scipy.ndimage.label(threshold_map, structure=[[1,1,1],[1,1,1],[1,1,1]])
    coordinates = numpy.mgrid[0:saliency_map.shape[0], 0:saliency_map.shape[1]].swapaxes(0, 2).swapaxes(0, 1)
    bounds = []
    for i in xrange(1, number_of_components + 1):
        component_idx = (components == i)
        items = coordinates[component_idx]
        if type == 'rect':
            min_bounds = numpy.min(items, axis=0)
            max_bounds = numpy.max(items, axis=0)
            x1 = min_bounds[1]
            y1 = min_bounds[0]
            x2 = max_bounds[1]
            y2 = max_bounds[0]
            assert((x1 <= x2) and (y1 <= y2) and (x2 <= saliency_map.shape[1]) and (y2 <= saliency_map.shape[0]))
            bounds.append((x1, y1, x2, y2))
        else:
            area = numpy.sum(component_idx)
            r = math.sqrt(float(area)) / 2.0
            center = numpy.mean(items, axis=0)
            x = center[1]
            y = center[0]
            assert((x < saliency_map.shape[1]) and (y < saliency_map.shape[0]))
            bounds.append((x, y, r))
    return bounds

def detect_objects(rgb_array, saliency_map, value='auto'):
    """
    Extracts salient objects from image.

    Parameters
    ----------
    rgb_image : ndarray
        Should be 3 channel RGB image with 8bit per channels
    saliency_map : ndarray
        Should be 2D-array with [0, 1] items.
    value : float or 'auto'
        Scalar should be in [0, 1].

    Returns
    -------
    out : list of ndarrays
        Extracted salient objects (sub-images of input image containing objects).
    """
    bounds = detect_bounds(saliency_map, value='auto')
    result = []
    for x1, y1, x2, y2 in bounds:
        object = rgb_array[y1:y2+1, x1:x2+1, :]
        result.append(object)
    return result

if __name__ == '__main__':
    print __doc__