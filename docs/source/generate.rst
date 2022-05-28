Generate
========

.. py:function:: angles_legend(images_amount, name, itype, step, mus, sigmas, amps, norm)

    Cоздание легенды распределения углов

        :param images_amount: int
        :param name: str
        :param itype: str
        :param step: int
        :param mus: float
        :param sigmas: float
        :param amps: float
        :param norm: int
        :return: str
		
.. py:function:: angles_approx_save(folder, images, names, types, step, save=True)

    Вычисление и сохранение распределения углов для всех фотографий одного образца

        :param folder: str path to dir
        :param images: ndarray uint8 [[image1_class1,image2_class1,..],[image1_class2,image2_class2,..]..]
        :param names: list str [class_name1,class_name2,..]
        :param types: list str [class_type1,class_type2,..]
        :param step: scalar int [0,N]
        :param save: bool
		
.. py:function:: beams_legend(name, itype, norm, k, angle, b, score, dist_step, dist_mean)

    Создание легенды для распределения длин полуосей

        :param name: str
        :param itype: str 
        :param norm: int
        :param k: float
        :param angle: float 
        :param b: float 
        :param score: float 
        :param dist_step: int 
        :param dist_mean: float
        :return: str
		
.. py:function:: diametr_approx_save(folder, images, names, types, step, pixel, start=2, end=-3, save=True, debug=False)

    Вычисление и сохранение распределения длин а- и б- полуосей и угла поворота эллипса для разных образцов

        :param folder: str
        :param images: ndarray uint8 [[image1_class1,image2_class1,..],[image1_class2,image2_class2,..]..]
        :param names: list str [class_name1,class_name2,..]
        :param types: list str [class_type1,class_type2,..]
        :param step: scalar int [0,N]
        :param pixel: float
        :param start: int
        :param end: int
        :param save: bool
        :param debug: bool
        :return: None