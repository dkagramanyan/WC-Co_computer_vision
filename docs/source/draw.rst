Draw
====

.. py:function:: draw_corners(image, corners, color=255)

    Наносит на изображение точки в местах, где есть углы списка corners

        :param image: ndarray (width, height, channels)
        :param corners: list (n_corners,2)
        :param color:  int
        :return: ndarray (width, height, channels)

.. py:function:: draw_edges(image, cnts, color=(50, 50, 50))

    Рисует на изображении линии по точкам контура cnts,  линии в стиле x^1->x^2,x^2->x^3 и тд

        :param image: ndarray (width, height, channels)
        :param cnts: ndarray (n_cnts,n,2)
        :param color: tuple (3,)
        :return: ndarray (width, height, channels)
		

.. py:function:: draw_tree(image, centres=False, leafs=False, nodes=False, bones=False)

    На вход подается бинаризованное изображение. Рисует на инвертированном изображении скелет: точки их центров, листьев, узлов и пикселей скелета

        :param img: ndarray (width, height)
        :param centres: Bool
        :param leafs: Bool
        :param nodes: Bool
        :param bones: Bool
        :return: ndarray (width, height, channels)
