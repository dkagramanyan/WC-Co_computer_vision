Fig
===


.. py:function::  line(point1, point2)

    Возвращает растровые координаты прямой между двумя точками при помощи алгоритма `Брезенхема <https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm>`_

	:param point1: tuple (int, int)
	:param point2: tuple (int, int)
	:return: ndarray (n_points,(x,y))
	
.. py:function::  rect(point1, point2)

    Возвращает растровые координаты прямоугольника ширины 2r, построеного между двумя точками. Не ясно зачем умножать в размерность на 2. 

        :param point1: tuple (int, int)
        :param point2: tuple (int, int)
        :param r: int
        :return: tuple (n_points, rect_diag*2,2 )