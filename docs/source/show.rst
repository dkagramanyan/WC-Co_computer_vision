Show
====

.. py:function:: img_show(image, N=20, cmap=plt.cm.nipy_spectral)

    Выводит изображение image

        :param image:  ndarray (height,width,channels)
        :param N: int
        :param cmap: plt cmap
        :return: None

.. py:function:: enclosing_ellipse_show(image, pos=0, tolerance=0.2, N=15)

    Выводит точки многоугольника с позиции pos и описанного вокруг него эллипса

        :param image: ndarray (height,width,channels)
        :param pos: int
        :param tolerance: foat, koef of ellipse compactness
        :param N: int
        :return: None