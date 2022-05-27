Morphology
==========


.. py:function::  kmeans_image(image, n_clusters)

    Кластеризует при помощи kmeans и возвращает изображение с нанесенными цветами кластеров

	:param image: ndarray (width, height, channels)
	:param n_clusters: int
	:return: (image, colors),colors - list of median colors of the clusters