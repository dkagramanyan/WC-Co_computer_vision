from scipy.spatial import distance
from scipy.stats import chisquare, kstest, entropy

from sklearn import metrics

import numpy as np
import os
import time

from lmfit.models import GaussianModel, Model
from sklearn.linear_model import LinearRegression
from sklearn import mixture, metrics

from shapely.geometry import Polygon

from scipy.stats.distributions import norm
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

from matplotlib import pyplot as plt
from matplotlib import cm

from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse

from scipy import ndimage as ndi
from scipy.spatial import distance
from numpy import linalg

import skimage
from skimage import io, transform
from skimage.draw import ellipse
from skimage.color import rgb2gray
from skimage import filters
from skimage.morphology import disk
from skimage import color

from PIL import Image, ImageDraw, ImageFilter, ImageOps
import copy
import cv2
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import distance_transform_edt as edt
from skimage.draw import ellipse
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
import logging
import glob
from radio_beam.commonbeam import getMinVolEllipse

from src.utils import grainPreprocess, grainShow, grainMark, grainDraw, grainApprox, grainStats, grainMorphology, \
    grainGenerate
from src.cfg import CfgAnglesNames, CfgBeamsNames, CfgDataset


class grainMarkTests():

    @classmethod
    def wc_co_angles_calculator_test(cls, images, step=2, N_tests=10, loss_thr=8):
        angles_dens = []

        for image in images:
            all_angles = grainMark.get_angles(image)
            _, _, dens_curve = grainStats.stats_preprocess(all_angles, step)
            norm = np.sum(dens_curve)
            dens = dens_curve / norm
            angles_dens.append(dens)

        names = np.array(
            ['Ultra_Co8', 'Ultra_Co11', 'Ultra_Co6_2', 'Ultra_Co15', 'Ultra_Co25'])

        for j, dens in enumerate(angles_dens):
            # cls_angles, angles_set, dens_curve = grainStats.stats_preprocess(angles, step)

            x = np.linspace(0, 361, dens.shape[0])

            true_loss = []
            true_loss_ks = []

            mus1_true = np.random.randint(100, 120, N_tests)
            mus2_true = np.random.randint(230, 250, N_tests)

            sigmas1_true = np.random.randint(25, 40, N_tests)
            sigmas2_true = np.random.randint(20, 35, N_tests)

            amps1_true = np.random.uniform(0.5, 4, N_tests)
            amps2_true = np.random.uniform(0.2, 2, N_tests)

            for i in range(N_tests):
                true_gauss = grainStats.gaussian_bimodal(x, mus1_true[i], mus2_true[i], sigmas1_true[i],
                                                         sigmas2_true[i], amps1_true[i], amps2_true[i])
                true_loss.append(cross_entropy(dens, true_gauss))
                true_loss_ks.append(kstest(dens, true_gauss)[0])

            true_loss_mean = np.mean(true_loss)
            true_loss_ks_mean = np.mean(true_loss_ks)

            if true_loss_mean < loss_thr:
                txt = ''
            else:
                txt = 'NOT'

            print(
                f'Test №{j + 1} {names[j]} {txt} passed, true={np.round(true_loss_mean)}, threshold={loss_thr}, ks test={np.round(true_loss_ks_mean, 2)}')
            plt.plot(np.linspace(0, 361, angles_dens[j].shape[0]), angles_dens[j])
            plt.show()


def cross_entropy(x, y):
    ''' SEE: https://en.wikipedia.org/wiki/Cross_entropy'''
    return entropy(x) + entropy(x, y)