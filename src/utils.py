import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib import cm

from lmfit.models import Model

from sklearn.cluster import KMeans

from shapely.geometry import Polygon

from radio_beam.commonbeam import getMinVolEllipse

from scipy import ndimage as ndi
from scipy.spatial import distance

from skimage import io
from skimage.measure import EllipseModel
from skimage.color import rgb2gray
from skimage import filters
from skimage.morphology import disk, skeletonize
from skimage.measure import approximate_polygon

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from sklearn.linear_model import LinearRegression

from scipy import ndimage

import copy
import cv2

from scipy.spatial import ConvexHull

import sys
import logging
from logging import StreamHandler, Formatter

from src.cfg import CfgAnglesNames, CfgBeamsNames, CfgDataset

handler = StreamHandler(stream=sys.stdout)
handler.setFormatter(Formatter(fmt='[%(asctime)s: %(levelname)s] %(message)s'))

logger = logging.getLogger(__name__)
logger.addHandler(handler)

file_path = os.getcwd() + '/utils.py'


class grainPreprocess():

    @classmethod
    def imdivide(cls, image: np.ndarray, h: int, side: str) -> np.ndarray:
        """
        :param image: ndarray (height,width,channels)
        :param h: int scalar
        :param side: str 'left'
        :return: ndarray (height,width/2,channels)
        """
        #
        # возвращает левую или правую часть полученного изображения
        #
        height, width = image.shape
        sides = {'left': 0, 'right': 1}
        shapes = [(0, height - h, 0, width // 2), (0, height - h, width // 2, width)]
        shape = shapes[sides[side]]

        return image[shape[0]:shape[1], shape[2]:shape[3]]

    @classmethod
    def combine(cls, image: np.ndarray, h: int, k=0.5) -> np.ndarray:
        """
        :param image: ndarray (height,width,channels)
        :param h: int scalar
        :param k: float scalar
        :return: ndarray (height,width/2,channels)
        """
        #
        #  накладывает левую и правые части изображения
        #  если k=1, то на выходе будет левая часть изображения, если k=0, то будет правая часть
        #
        left_img = cls.imdivide(image, h, 'left')
        right_img = cls.imdivide(image, h, 'right')

        l = k
        r = 1 - l
        gray = np.array(left_img) * l
        gray += np.array(right_img) * r
        gray = gray.astype('uint8')
        img = rgb2gray(gray)
        return img

    @classmethod
    def do_otsu(cls, img: np.ndarray) -> np.ndarray:
        """
        :param img: ndarray (height,width,channels)
        :return: ndarray (height,width), Boolean
        """
        #
        # бинаризация отсу
        #
        global_thresh = filters.threshold_otsu(img)
        binary_global = img > global_thresh

        return binary_global

    @classmethod
    def image_preprocess(cls, image: np.ndarray, h=135, k=1) -> np.ndarray:
        """
        :param image: ndarray (height,width,channels)
        :param h: int scalar
        :param k: float scalar
        :return: ndarray (height,width)
        """
        #
        # комбинация медианного фильтра, биноризации и гражиента
        # у зерен значение пикселя - 0, у регионов связ. в-ва - 1,а у их границы - 2
        #
        combined = cls.combine(image, h, k)
        denoised = filters.rank.median(combined, disk(3))
        binary = cls.do_otsu(denoised).astype('uint8')
        grad = abs(filters.rank.gradient(binary, disk(1))).astype('uint8')
        bin_grad = 1 - binary + grad
        new_image = (bin_grad > 0).astype('uint8') * 255

        return new_image

    @classmethod
    def image_preprocess_kmeans(cls, image: np.ndarray, h=135, k=1, n_clusters=3, pos=1) -> np.ndarray:
        """
        :param image: array (height,width,channels)
        :param h: int scalar
        :param k: float scalar
        :param n_clusters: int scalar
        :param pos: int scalar, cluster index
        :return: ndarray (height,width)
        """
        #
        # выделение границ при помощи кластеризации 
        # и выравнивание шума медианным фильтром
        # pos отвечает за выбор кластера, который будет отображен на возвращенном изображении
        #
        combined = cls.combine(image, h, k)

        clustered, colors = grainMorphology.kmeans_image(combined, n_clusters)
        cluster = clustered == colors[pos]
        cluster = np.array(cluster * 255, dtype='uint8')

        new_image = filters.median(cluster, disk(2))
        return new_image

    @classmethod
    def read_preprocess_data(cls, images_dir, images_num_per_class=100, preprocess=False, save=False, crop=False, h=135,
                             save_name='all_images.npy'):
        folders_names = os.listdir(images_dir)
        images_paths_raw = [os.listdir(images_dir + '/' + folder) for folder in folders_names]

        images_paths = []
        all_images = []
        for i, folder in enumerate(images_paths_raw):
            images_paths.append([])
            for image_path in folder:
                images_paths[i].append(images_dir + '/' + folders_names[i] + '/' + image_path)

        for images_folder in images_paths:
            images = [io.imread(name) for i, name in enumerate(images_folder) if i < images_num_per_class]
            if preprocess:
                images = [grainPreprocess.image_preprocess(image) for image in images]
            if not preprocess and crop:
                images = [grainPreprocess.combine(image, h) for image in images]

            all_images.append(images)
        if save:
            np.save(save_name, all_images)
        return all_images

    @classmethod
    def tiff2jpg(cls, folder_path, start_name=0, stop_name=-4, new_folder_path='resized'):

        folders = os.listdir(folder_path)

        if not os.path.exists(new_folder_path):
            os.mkdir(new_folder_path)

        for folder in folders:
            if not os.path.exists(new_folder_path + '/' + folder):
                os.mkdir(new_folder_path + '/' + folder)

        for i, folder in enumerate(folders):
            images_names = os.listdir(folder_path + '/' + folder)
            for i, name in enumerate(images_names):
                if 'hdr' not in name:
                    img = io.imread(folder_path + '/' + folder + '/' + name)
                    img = (img / 255).astype('uint8')

                    io.imsave(new_folder_path + '/' + folder + '/' + name[start_name:stop_name] + '.jpg', img)


class grainMorphology():

    @classmethod
    def kmeans_image(cls, image, n_clusters=3):
        #
        # кластеризует при помощи kmeans
        # и возвращает изображение с нанесенными цветами кластеров
        #
        img = image.copy()

        size = img.shape
        img = img.reshape(-1, 1)

        model = KMeans(n_clusters=n_clusters)
        clusters = model.fit_predict(img)

        colors = []
        for i in range(n_clusters):
            color = np.median(img[clusters == i])  # медианное значение пикселей у кластера
            img[clusters == i] = color
            colors.append(int(color))

        img = img.reshape(size)
        colors.sort()

        return img, colors


class grainFig():

    @classmethod
    def line(cls, point1, point2):
        #
        # возвращает растровые координаты прямой между двумя точками 
        #
        line = []

        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]

        dx = x2 - x1
        dy = y2 - y1

        sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
        sign_y = 1 if dy > 0 else -1 if dy < 0 else 0

        if dx < 0: dx = -dx
        if dy < 0: dy = -dy

        if dx > dy:
            pdx, pdy = sign_x, 0
            es, el = dy, dx
        else:
            pdx, pdy = 0, sign_y
            es, el = dx, dy

        x, y = x1, y1
        error, t = el / 2, 0

        line.append((x, y))

        while t < el:
            error -= es
            if error < 0:
                error += el
                x += sign_x
                y += sign_y
            else:
                x += pdx
                y += pdy
            t += 1
            line.append((x, y))
        return np.array(line).astype('int')

    @classmethod
    def rect(cls, point1, point2, r):
        #
        # возвращает растровые координаты прямоугольника ширины 2r,
        # построеного между двумя точками 
        #
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]

        l1, l2 = (x2 - x1), (y2 - y1)

        l_len = (l1 ** 2 + l2 ** 2) ** 0.5
        l_len = int(l_len)

        a = (x1 - r * l2 / l_len), (y1 + r * l1 / l_len)
        b = (x1 + r * l2 / l_len), (y1 - r * l1 / l_len)

        side = cls.line(a, b)

        # a -> c
        lines = np.zeros((side.shape[0], l_len * 2, 2), dtype='int64')

        for i, left_point in enumerate(side):
            right_point = (left_point[0] + l1), (left_point[1] + l2)
            line_points = cls.line(left_point, right_point)
            for j, point in enumerate(line_points):
                lines[i, j] = point

        return lines


class grainMark():
    @classmethod
    def mark_corners_and_classes(cls, image, max_num=100000, sens=0.1, max_dist=1):
        #
        # НЕТ ГАРАНТИИ РАБОТЫ
        #
        corners = cv2.goodFeaturesToTrack(image, max_num, sens, max_dist)
        corners = np.int0(corners)
        x = copy.copy(corners[:, 0, 1])
        y = copy.copy(corners[:, 0, 0])
        corners[:, 0, 0], corners[:, 0, 1] = x, y

        classes = filters.rank.gradient(image, disk(1)) < 250
        classes, num = ndi.label(classes)
        return corners, classes, num

    @classmethod
    def mean_pixel(cls, image, point1, point2, r):

        val2, num2 = cls.draw_rect(image, point2, point1, r)
        val = val1 + val2
        num = num1 + num2

        if num != 0 and val != 0:
            mean = (val / num) / 255
            dist = distance.euclidean(point1, point2)
        else:
            mean = 1
            dist = 1
        return mean, dist

    @classmethod
    def get_row_contours(cls, image):
        """
        :param image: ndarray (width, height,3)
        :return: list (N_contours,M_points,2)
            where ndarray (M_points,2)
        """
        #
        # возвращает набор точек контуров 
        #
        edges = cv2.Canny(image, 0, 255, L2gradient=False)

        # направление обхода контура по часовой стрелке
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for i, cnt in enumerate(contours):
            contours[i] = cnt[:, 0]
        return contours

    @classmethod
    def get_contours(cls, image, tol=3):
        """
        :param tol:
        :param image: ndarray (width, height,3)
        :return: list (N_contours,M_points,2)
            where ndarray (M_points,2)
        """
        #
        # уменьшение количества точек контура при помощи алгоритма Дугласа-Пекера
        #
        contours = cls.get_row_contours(image)

        new_contours = []
        for j, cnt in enumerate(contours):
            if len(cnt) > 2:
                coords = approximate_polygon(cnt, tolerance=tol)
                new_contours.append(coords)
            else:
                continue

        return new_contours

    @classmethod
    def get_angles(cls, image, thr=5):
        #
        # считаем углы с направлением обхода контура против часовой стрелки, углы >180 градусов учитываются
        #
        approx = cls.get_contours(image, tol=4)

        # вычисление угла
        angles = []
        for k, cnt in enumerate(approx):
            if len(cnt) > 2:
                for i, point in enumerate(cnt[:-1]):
                    point1 = cnt[i - 1]
                    point2 = cnt[i]
                    point3 = cnt[i + 1]
                    x1, y1 = point1[1], point1[0]
                    x2, y2 = point2[1], point2[0]
                    x3, y3 = point3[1], point3[0]
                    # убирает контуры у границ

                    if abs(x2 - image.shape[0] - 1) > thr and \
                            abs(y2 - image.shape[1] - 1) > thr and \
                            x2 > thr and y2 > thr:
                        v1 = np.array((x1 - x2, y1 - y2)).reshape(1, 2)
                        v2 = np.array((x3 - x2, y3 - y2)).reshape(1, 2)

                        dot = np.dot(v1[0], v2[0])
                        dist1 = np.linalg.norm(v1[0])
                        dist2 = np.linalg.norm(v2[0])
                        cos = dot / (dist1 * dist2)

                        v = np.concatenate([v1, v2])
                        det = np.linalg.det(v)

                        if abs(cos) < 1:
                            ang = int(np.arccos(cos) * 180 / np.pi)
                            if det < 0:
                                angles.append(ang)
                            else:
                                angles.append(360 - ang)
                        else:
                            if det < 0:
                                angles.append(360)
                            else:
                                angles.append(0)

        return np.array(angles)

    @classmethod
    def get_mvee_params(cls, image, tol=0.2):
        """
        :param image:
        :param tol:
        :return: ndarray (n_angles), radian
        """
        #
        # возвращает полуоси и угол поворота фигуры minimal volume enclosing ellipsoid,
        # которая ограничивает исходные точки контура эллипсом
        # 
        approx = grainMark.get_row_contours(image)
        a_beams = []
        b_beams = []
        angles = []
        centroids = []

        for i, cnt in enumerate(approx):
            if len(cnt) > 2:
                cnt = np.array(cnt)
                polygon = Polygon(cnt)

                x_centroid, y_centroid = polygon.centroid.coords[0]
                points = cnt - (x_centroid, y_centroid)

                x_norm, y_norm = points.mean(axis=0)
                points = (points - (x_norm, y_norm))
                data = getMinVolEllipse(points, tol)

                xc, yc = data[0][0]
                a, b = data[1]
                sin = data[2][0][1]
                angle = -np.arcsin(sin)

                a_beams.append(a)
                b_beams.append(b)
                angles.append(angle)
                centroids.append([x_centroid + x_norm, y_centroid + y_norm])

        a_beams = np.array(a_beams, dtype='int32')
        b_beams = np.array(b_beams, dtype='int32')
        angles = np.array(angles, dtype='float32')
        centroids = np.array(centroids, dtype='int32')

        return a_beams, b_beams, angles, centroids

    @classmethod
    def skeletons_coords(cls, image):
        #
        # на вход подается бинаризованное изображение
        # создает массив индивидуальных скелетов
        # пикселю скелета дается класс, на координатах которого он находится
        # координаты класса определяются ndi.label
        #
        skeleton = np.array(skeletonize(image))
        labels, classes_num = ndimage.label(image)

        bones = [[] for i in range(classes_num + 1)]

        for i in range(skeleton.shape[0]):
            for j in range(skeleton.shape[1]):
                if skeleton[i, j]:
                    label = labels[i, j]
                    bones[label].append((i, j))
        return bones


class grainShow():

    @classmethod
    def img_show(cls, image, N=20, cmap=plt.cm.nipy_spectral):
        #
        # выводит изображение image
        #

        plt.figure(figsize=(N, N))
        plt.axis('off')
        plt.imshow(image, cmap=cmap)
        plt.show()

    @classmethod
    def enclosing_ellipse_show(cls, image, pos=0, tolerance=0.2, N=15):
        #
        # рисует график точек многоугольника и описанного эллипса
        #
        a_beams, b_beams, angles, cetroids = grainMark.get_mvee_params(image, tolerance)
        approx = grainMark.get_row_contours(image)

        a = a_beams[pos]
        b = b_beams[pos]
        angle = angles[pos]
        print('полуось а ', a)
        print('полуось b ', b)
        print('угол поворота ', round(angle, 3), ' радиан')

        cnt = np.array(approx[pos])

        xp = cnt[:, 0]
        yp = cnt[:, 1]
        xc = cetroids[pos, 0]
        yc = cetroids[pos, 1]

        x, y = grainStats.ellipse(a, b, angle)

        plt.figure(figsize=(N, N))
        plt.plot(xp - xc, yp - yc)
        plt.scatter(0, 0)
        plt.plot(x, y)

        plt.show()


class grainDraw():
    @classmethod
    def draw_corners(cls, image, corners, color=255):
        #
        # НЕТ ГАРАНТИИ РАБОТЫ
        #
        image = copy.copy(image)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(image, (x, y), 3, color, -1)

        return image

    @classmethod
    def draw_edges(cls, image, cnts, color=(50, 50, 50)):
        #
        # рисует на изображении линии по точкам контура cnts
        # линии в стиле x^1->x^2,x^2->x^3 и тд
        #
        new_image = copy.copy(image)
        im = Image.fromarray(np.uint8(cm.gist_earth(new_image) * 255))
        draw = ImageDraw.Draw(im)

        for j, cnt in enumerate(cnts):
            if len(cnt) > 1:
                point = cnt[0]
                x1, y1 = point[1], point[0]
                r = 4

                for i, point2 in enumerate(cnt):
                    p2 = point2

                    x2, y2 = p2[1], p2[0]

                    draw.ellipse((y2 - r, x2 - r, y2 + r, x2 + r), fill=color, width=5)
                    draw.line((y1, x1, y2, x2), fill=(100, 100, 100), width=4)
                    x1, y1 = x2, y2

            else:
                continue

        img = np.array(im)

        return img

    @classmethod
    def draw_tree(cls, img, centres=False, leafs=False, nodes=False, bones=False):
        #
        # на вход подается биноризованное изображение
        # рисует на инвертированном изображении
        # скелет и точки центров, листьев, узлов и пикселей скелета
        #

        image = img.copy() / 255

        skeleton = np.array(skeletonize(image)) * 255
        im = 1 - image + skeleton
        im = Image.fromarray(np.uint8(cm.gist_earth(im) * 255))
        draw = ImageDraw.Draw(im)

        if bones:
            for j, bone in enumerate(bones):
                for i, point in enumerate(bone):
                    x2, y2 = point
                    r = 1
                    draw.ellipse((y2 - r, x2 - r, y2 + r, x2 + r), fill=(89, 34, 0), width=5)

        if centres:
            for j, point in enumerate(centres):
                x2, y2 = point
                r = 2
                draw.ellipse((y2 - r, x2 - r, y2 + r, x2 + r), fill=(255, 0, 0), width=5)

        if leafs:
            for j, leaf in enumerate(leafs):
                for i, point in enumerate(leaf):
                    x2, y2 = point
                    r = 2
                    draw.ellipse((y2 - r, x2 - r, y2 + r, x2 + r), fill=(0, 255, 0), width=5)
        if nodes:
            for j, node in enumerate(nodes):
                for i, point in enumerate(node):
                    x2, y2 = point
                    r = 2
                    draw.ellipse((y2 - r, x2 - r, y2 + r, x2 + r), fill=(0, 0, 255), width=10)

        return np.array(im)


class grainStats():
    @classmethod
    def kernel_points(cls, image, point, step=1):
        #
        # возвращает координаты пикселей матрицы,
        # центр которой это point
        #
        x, y = point
        coords = []
        for xi in range(x - step, x + step + 1):
            for yi in range(y - step, y + step + 1):
                if xi < image.shape[0] and yi < image.shape[1]:
                    coords.append((xi, yi))
        return coords

    @classmethod
    def stats_preprocess(cls, array, step):
        #
        # приведение углов к кратости, например 0,step,2*step и тд
        #
        array_copy = array.copy()

        for i, a in enumerate(array_copy):
            while array_copy[i] % step != 0:
                array_copy[i] += 1

        array_copy_set = np.sort(np.array(list(set(array_copy))))
        dens_curve = []
        for arr in array_copy_set:
            num = 0
            for ar in array_copy:
                if arr == ar:
                    num += 1
            dens_curve.append(num)
        return np.array(array_copy), array_copy_set, np.array(dens_curve)

    @classmethod
    def gaussian(cls, x, mu, sigma, amp=1):
        #
        # возвращает нормальную фунцию по заданным параметрам
        #
        return np.array((amp / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))

    @classmethod
    def gaussian_bimodal(cls, x, mu1, mu2, sigma1, sigma2, amp1=1, amp2=1):
        #
        # возвращает бимодальную нормальную фунцию по заданным параметрам
        #
        return cls.gaussian(x, mu1, sigma1, amp1) + cls.gaussian(x, mu2, sigma2, amp2)

    @classmethod
    def gaussian_termodal(cls, x, mu1, mu2, mu3, sigma1, sigma2, sigma3, amp1=1, amp2=1, amp3=1):
        #
        # возвращает термодальную нормальную фунцию по заданным параметрам
        #
        return cls.gaussian(x, mu1, sigma1, amp1) + cls.gaussian(x, mu2, sigma2, amp2) + cls.gaussian(x, mu3, sigma3,
                                                                                                      amp3)

    @classmethod
    def ellipse(cls, a, b, angle, xc=0, yc=0, num=50):
        #
        #  возвращает координаты эллипса, построенного по заданным параметрам
        #  по умолчанию центр (0,0)
        #  угол в радианах, уменьшение угла обозначает поворот эллипса по часовой стрелке
        #
        xy = EllipseModel().predict_xy(np.linspace(0, 2 * np.pi, num),
                                       params=(xc, yc, a, b, angle))
        return xy[:, 0], xy[:, 1]


class grainApprox():

    @classmethod
    def gaussian_fit(cls, x, y, mu=1, sigma=1, amp=1):
        #
        # аппроксимация заданных точек нормальной функцией
        #
        gmodel = Model(grainStats.gaussian)
        res = gmodel.fit(y, x=x, mu=mu, sigma=sigma, amp=amp)

        mu = res.params['mu'].value
        sigma = res.params['sigma'].value
        amp = res.params['amp'].value

        return mu, sigma, amp

    @classmethod
    def gaussian_fit_bimodal(cls, x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1):
        #
        # аппроксимация заданных точек бимодальной нормальной функцией
        #
        gmodel = Model(grainStats.gaussian_bimodal)
        res = gmodel.fit(y, x=x, mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2, amp1=amp1, amp2=amp2)

        mus = [res.params['mu1'].value, res.params['mu2'].value]
        sigmas = [res.params['sigma1'].value, res.params['sigma2'].value]
        amps = [res.params['amp1'].value, res.params['amp2'].value]

        return mus, sigmas, amps

    @classmethod
    def gaussian_fit_termodal(cls, x, y, mu1=10, mu2=100, mu3=240, sigma1=10, sigma2=30, sigma3=30, amp1=1, amp2=1,
                              amp3=1):
        #
        # аппроксимация заданных точек термодальной нормальной функцией
        #
        gmodel = Model(grainStats.gaussian_termodal)
        res = gmodel.fit(y, x=x, mu1=mu1, mu2=mu2, mu3=mu3, sigma1=sigma1, sigma2=sigma2, sigma3=sigma3, amp1=amp1,
                         amp2=amp2, amp3=amp3)

        mus = [res.params['mu1'].value, res.params['mu2'].value, res.params['mu3'].value]
        sigmas = [res.params['sigma1'].value, res.params['sigma2'].value, res.params['sigma3'].value]
        amps = [res.params['amp1'].value, res.params['amp2'].value, res.params['amp3'].value]

        return mus, sigmas, amps

    @classmethod
    def lin_regr_approx(cls, x, y):
        #
        # аппроксимация распределения линейной функцией
        # и создание графика по параметрам распределения
        #
        x_pred = np.linspace(x.min(axis=0), x.max(axis=0), 50)

        reg = LinearRegression().fit(x, y)
        y_pred = reg.predict(x_pred)

        k = reg.coef_[0][0]
        b = reg.predict([[0]])[0][0]

        angle = np.rad2deg(np.arctan(k))
        score = reg.score(x, y)

        return (x_pred, y_pred), k, b, angle, score

    @classmethod
    def bimodal_gauss_approx(cls, x, y):
        #
        # аппроксимация распределения бимодальным гауссом
        #

        mus, sigmas, amps = cls.gaussian_fit_bimodal(x, y)

        x_gauss = np.arange(0, 361)
        y_gauss = grainStats.gaussian_bimodal(x_gauss, mus[0], mus[1], sigmas[0], sigmas[1], amps[0], amps[1])

        return (x_gauss, y_gauss), mus, sigmas, amps


class grainGenerate():
    @classmethod
    def angles_legend(cls, images_amount, name, itype, step, mus, sigmas, amps, norm, ):
        #
        # создание легенды распределения углов
        #

        mu1 = round(mus[0], 2)
        sigma1 = round(sigmas[0], 2)
        amp1 = round(amps[0], 2)

        mu2 = round(mus[1], 2)
        sigma2 = round(sigmas[1], 2)
        amp2 = round(amps[1], 2)

        val = round(norm, 4)

        border = '--------------\n'
        total_number = '\n количество углов ' + str(val)
        images_number = '\n количество снимков ' + str(images_amount)
        text_angle = '\n шаг угла ' + str(step) + ' градусов'

        moda1 = '\n mu1 = ' + str(mu1) + ' sigma1 = ' + str(sigma1) + ' amp1 = ' + str(amp1)
        moda2 = '\n mu2 = ' + str(mu2) + ' sigma2 = ' + str(sigma2) + ' amp2 = ' + str(amp2)

        legend = border + name + ' ' + itype + total_number + images_number + text_angle + moda1 + moda2

        return legend

    @classmethod
    def angles_approx_save(cls, folder, images, names, types, step, save=True):
        """
        :param folder: str path to dir
        :param images: ndarray uint8 [[image1_class1,image2_class1,..],[image1_class2,image2_class2,..]..]
        :param names: list str [class_name1,class_name2,..]
        :param types: list str [class_type1,class_type2,..]
        :param step: scalar int [0,N]
        :param save: bool
        :return: ndarray uint8 (n_classes,n_samples, height, width)
        """
        #
        # вывод распределения углов для всех фотографий одного образца
        #

        texts = []
        xy_scatter = []
        xy_gauss = []
        xy_gauss_data = []

        if not os.path.exists(folder):
            os.mkdir(folder)

        for i, images_list in enumerate(images):
            all_original_angles = []

            for j, image in enumerate(images_list):
                original_angles = grainMark.get_angles(image)

                for angle in original_angles:
                    all_original_angles.append(angle)

            angles, angles_set, dens_curve = grainStats.stats_preprocess(all_original_angles, step)

            x = angles_set.astype(np.float64)
            y = dens_curve

            norm = np.sum(y)
            y = y / norm

            (x_gauss, y_gauss), mus, sigmas, amps = grainApprox.bimodal_gauss_approx(x, y)

            text = grainGenerate.angles_legend(len(images_list), names[i], types[i], step, mus, sigmas, amps, norm)

            xy_gauss.append((x_gauss, y_gauss))
            xy_scatter.append((x, y))
            xy_gauss_data.append((
                (mus[0], sigmas[0], amps[0]),
                (mus[1], sigmas[1], amps[1])
            ))

            texts.append(text)

        if save:
            np.save(f'{folder}/' + CfgAnglesNames.values + f'{step}.npy', np.array(xy_scatter, dtype=object))
            np.save(f'{folder}/' + CfgAnglesNames.approx + f'{step}.npy', np.array(xy_gauss))
            np.save(f'{folder}/' + CfgAnglesNames.approx_data + f'{step}.npy', np.array(xy_gauss_data))
            np.save(f'{folder}/' + CfgAnglesNames.legend + f'{step}.npy', np.array(texts))

    @classmethod
    def beams_legend(cls, name, itype, norm, k, angle, b, score, dist_step, dist_mean):
        #
        # создание легенды для распределения длин полуосей
        #
        border = '--------------'
        tp = '\n ' + name + ' тип ' + itype
        num = '\n регионы Co ' + str(norm) + ' шт'
        lin_k = '\n k наклона ' + str(round((k), 3)) + ' сдвиг b ' + str(round(b, 3))
        lin_k_angle = '\n угол наклона $' + str(round(angle, 3)) + '^{\circ}$'
        acc = '\n точность ' + str(round(score, 2))
        text_step = '\n шаг длины ' + str(dist_step) + '$ мкм$'
        mean_text = '\n средняя длина ' + str(round(dist_mean, 2))
        legend = border + tp + lin_k + lin_k_angle + acc + num + text_step + mean_text

        return legend

    @classmethod
    def diametr_approx_save(cls, folder, images, names, types, step, pixel, start=2, end=-3, save=True, debug=False):
        #
        # вывод распределения длин а- и б- полуосей для разных образцов
        #
        texts = []
        xy_scatter = []
        xy_linear = []
        xy_linear_data = []

        for i, images_list in enumerate(images):
            all_a_beams = []
            all_b_beams = []

            for j, image in enumerate(images_list):
                try:
                    a_beams, b_beams, angles, cetroids = grainMark.get_mvee_params(image, 0.2)
                    for k in range(len(a_beams)):
                        all_a_beams.append(a_beams[k])
                        all_b_beams.append(b_beams[k])
                except Exception:
                    logger.warning(f'{file_path} error i={i} j={j}, singularity matrix error, no reason why',
                                   exc_info=debug)

            distances1, dist1_set, dens1_curve = grainStats.stats_preprocess(all_a_beams, step)
            distances2, dist2_set, dens2_curve = grainStats.stats_preprocess(all_b_beams, step)

            angles, angles_set, angles_dens_curve = grainStats.stats_preprocess(np.rad2deg(angles).astype('int32'),
                                                                                step=5)

            norm1 = round(np.sum(dens1_curve), 6)
            norm2 = round(np.sum(dens2_curve), 6)

            x1 = np.array([dist1_set]).reshape(-1, 1) * pixel
            y1 = np.log([dens1_curve / norm1]).reshape(-1, 1)

            x2 = np.array([dist2_set]).reshape(-1, 1) * pixel
            y2 = np.log([dens2_curve / norm2]).reshape(-1, 1)

            x1 = x1[start:end]
            y1 = y1[start:end]

            x2 = x2[start:end]
            y2 = y2[start:end]

            (x_pred1, y_pred1), k1, b1, angle1, score1 = grainApprox.lin_regr_approx(x1, y1)
            (x_pred2, y_pred2), k2, b2, angle2, score2 = grainApprox.lin_regr_approx(x2, y2)

            dist_step = pixel * step

            legend1 = cls.beams_legend(names[i], types[i], norm1, k1, angle1, b1, score1, dist_step,
                                       distances1.mean() * pixel)
            legend2 = cls.beams_legend(names[i], types[i], norm2, k2, angle2, b2, score2, dist_step,
                                       distances2.mean() * pixel)

            texts.append([legend1, legend2])
            xy_scatter.append([(x1, y1), (x2, y2)])
            xy_linear.append((
                (x_pred1, y_pred1),
                (x_pred2, y_pred2)
            ))
            xy_linear_data.append((
                (k1, b1, angle1, score1),
                (k2, b2, angle2, score2)
            ))

        if save:
            np.save(f'{folder}/' + CfgBeamsNames.values + f'{step}.npy', np.array(xy_scatter, dtype=object))
            np.save(f'{folder}/' + CfgBeamsNames.approx + f'{step}.npy', np.array(xy_linear))
            np.save(f'{folder}/' + CfgBeamsNames.approx_data + f'{step}.npy', np.array(xy_linear_data))
            np.save(f'{folder}/' + CfgBeamsNames.legend + f'{step}.npy', np.array(texts))
