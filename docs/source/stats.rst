Stats
=====


.. py:function:: kernel_points(image, point, step=1)

    Возвращает координаты пикселей квадратной матрицы шириной 2*step, центр которой это point

        :param image: ndarray (width, height)
        :param point: tuple (2,)
        :param step: int
        :return: tuple (n_points,2)
		
.. py:function:: stats_preprocess(array, step)

    Приведение углов к кратости, например 0,step,2*step и тд

        :param array: list, ndarray (n,)
        :param step: int
        :return: array_copy, array_copy_set, dens_curve

.. py:function:: gaussian(x, mu, sigma, amp=1)

    Наносит на изображение точки в местах, где есть углы списка corners

        :param x: list (n,)
        :param mu: float
        :param sigma: float
        :param amp: float
        :return: list (n,)
		
.. py:function:: gaussian_bimodal(x, mu1, mu2, sigma1, sigma2, amp1=1, amp2=1)

    Возвращает бимодальную нормальную фунцию по заданным параметрам

        :param x: list (n,)
        :param mu1: float
        :param mu2: float
        :param sigma1: float
        :param sigma2: float
        :param amp1: float
        :param amp2: float
        :return: list (n,)
		
.. py:function::  gaussian_termodal(x, mu1, mu2, mu3, sigma1, sigma2, sigma3, amp1=1, amp2=1, amp3=1)

    Возвращает термодальную нормальную фунцию по заданным параметрам

        :param x: list (n,)
        :param mu1: float
        :param mu2: float
        :param mu3: float
        :param sigma1: float
        :param sigma2: float
        :param sigma3: float
        :param amp1: float
        :param amp2: float
        :param amp3: float
        :return: list (n,)
		
.. py:function:: ellipse(a, b, angle, xc=0, yc=0, num=50)

    Возвращает координаты эллипса, построенного по заданным параметрам. По умолчанию центр (0,0). Угол в радианах, уменьшение угла обозначает поворот эллипса по часовой стрелке

        :param a: float
        :param b: float
        :param angle: float, rad
        :param xc: float, center coord x
        :param yc: float, center coord y
        :param num: int, number of ellipse points
        :return: tuple (num, 2)