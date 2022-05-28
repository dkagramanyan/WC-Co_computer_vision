Approximation
=============

		
.. py:function:: gaussian_fit(x, y, mu=1, sigma=1, amp=1)

    Аппроксимация заданных точек нормальной функцией

        :param x: list (n,)
        :param y: list (n,)
        :param mu: float
        :param sigma: float
        :param amp: float
        :return: mus, sigmas, amps
		
.. py:function:: gaussian_fit_bimodal(x, y, mu1=100, mu2=240, sigma1=30, sigma2=30, amp1=1, amp2=1)

    Аппроксимация заданных точек бимодальной нормальной функцией

        :param x: list (n,)
        :param y: list (n,)
        :param mu1: float
        :param mu2: float
        :param sigma1: float
        :param sigma2: float
        :param amp1: float
        :param amp2: float
        :return: mus, sigmas, amps
		
.. py:function:: gaussian_fit_termodal(x, y, mu1=10, mu2=100, mu3=240, sigma1=10, sigma2=30, sigma3=30, amp1=1, amp2=1, amp3=1)

    Аппроксимация заданных точек термодальной нормальной функцией

        :param x: list (n,)
        :param y: list (n,)
        :param mu1: float
        :param mu2: float
        :param mu3: float
        :param sigma1: float
        :param sigma2: float
        :param sigma3: float
        :param amp1: float
        :param amp2: float
        :param amp3: float
        :return: mus, sigmas, amps
		
.. py:function:: lin_regr_approx(x, y)

    Аппроксимация распределения линейной функцией и создание графика по параметрам распределения

        :param x: list (n,)
        :param y: list (n,)
        :return: (x_pred, y_pred), k, b, angle, score
		
.. py:function:: bimodal_gauss_approx(x, y)

    Аппроксимация распределения бимодальным гауссом

        :param x: list (n,)
        :param y: list (n,)
        :return: (x_gauss, y_gauss), mus, sigmas, amps