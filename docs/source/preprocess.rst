Предобработка изображений
=========================

Описанные инструменты разработаны для обработки SEM снимков.


Для явного выделения границ фаз WC/Co используется последовательное применение следующиих алгоритом

1) выбирается сторона снимка. По умолчанию использвуется снимок в отраженных электронах

2) изображение сглаживается медианным фильтром для подавления шумов и выравнивания цветового распределения пикселей

3) слаженное изображение бинаризуется при помощи метода Отсу

4) от бинаризованного изображения берется градиент. Он явно показывает переходы вас WC/Co

5) бинаризованное изобржение инвертируется и к нему добавляется карта градиентов, полученная в п.4

Полная обработка изображения выглядит следующим образом:

        preproc_image=1-Otsu(median_filter(image))+grad(Otsu(median_filter(image)))

Значения пикселей по классам:

* 0 - зерно WC

* 127 - регион Co

* 254 - граница региона Co, смежного с зернами WC. Толщина границы - 1 пиксель

Обработка одного снимка

        img=io.imread(img_path)
        img=grainPreprocess.image_preprocess(img,h,k)

Обработка всего датасета снимков

        all_images=grainPreprocess.read_preprocess_data(images_folder_path, images_num_per_class=150,  preprocess=True, save=True)
		

Расположение снимков
--------------------

Расположение исходных снимков и предобработанных снимков должно выглядеть следующим образом

::

    project
    │
    └───images_folder
       │
       └───class1_images
       │       image1
       │       image2
       │       ...
       └───class2_images
       │       image1
       │       image2
       │       ...
       └───class3_images
       │       image1
       │       image2
       │       ...



Класс grainPreprocess
---------------------

.. py:function::  imdivide(image, h, side)

   Разделяет входное изображение по середине и возвращает левую или правую часть

	:param image: ndarray (height,width,channels)
	:param h: int scalar
	:param side: float scalar
	:return: ndarray (height,width/2,channels)
	
	
.. py:function::  combine(image, h, k=0.5)

   Накладывает левую и правые части изображения. Если k=1, то на выходе будет левая часть изображения, если k=0, то будет правая часть

	:param image: ndarray (height,width,channels)
	:param h: int scalar
	:param k: float scalar
	:return: ndarray (height,width/2,channels)
	
.. py:function::  do_otsu(image)

   Бинаризация Отсу

	:param img: ndarray (height,width,channels)
	:return: ndarray (height,width), Boolean
	

.. py:function::  image_preprocess(image)

	Комбинация медианного фильтра, биноризации и градиента. У зерен значение пикселя - 0, у регионов связ. в-ва - 127,а у их границы - 254.
	Обраотанное изображение получается следующим образом: 1-Otsu(median_filter(image))+grad(Otsu(median_filter(image)))

	
	 :param img: ndarray (height,width,channels)
	 :return: ndarray (height,width,1)
	 
.. py:function::  read_preprocess_data(images_dir, max_images_num_per_class=100, preprocess=False, save=False, crop_bottom=False, h=135, resize=True, resize_shape=None, save_name='all_images.npy')

   Считывание всего датасета, обработка и сохрание в .npy файл

        :param images_dir: str
        :param max_images_num_per_class: int
        :param preprocess: Bool
        :param save: Bool
        :param crop_bottom: Bool
        :param h: int
        :param resize: Bool
        :param resize_shape: tuple (width, height, channels)
        :param save_name: str
        :return: ndarray (n_classes, n_images_per_class, width, height, channels)
		
.. py:function::  tiff2jpg( folder_path, start_name=0, stop_name=-4, new_folder_path='resized')

   Переводит из tiff 2^16 в jpg 2^8 бит

        :param folder_path: str
        :param start_name: int
        :param stop_name: int
        :param new_folder_path: str
        :return: None
	
.. py:function::  get_example_images()

   Скачивает из контейнера s3 по 1 снимку каждого образца

        :return: ndarray [[img1],[img2]..]

.. py:function:: image_preprocess_kmeans(image, h=135, k=1, n_clusters=3, pos=1)

   Выделение границ при помощи кластеризации и выравнивание шума медианным фильтром. Подходит только 
   для смазанных фотограий, где границы у объектов достатчно широкие. 
   Pos отвечает за выбор кластера, который будет отображен на возвращенном изображении

        :param image: array (height,width,channels)
        :param h: int scalar
        :param k: float scalar
        :param n_clusters: int scalar
        :param pos: int scalar, cluster index
        :return: ndarray (height,width)
	
	
	
	


