import numpy as np


class CfgAnglesNames():
    values = 'angles_scatter_xy_step_'
    approx = 'angles_gauss_xy_step_'
    approx_data = 'angles_gauss_data_step_'
    legend = 'angles_legend_step_'


class CfgBeamsNames():
    values = 'beams_scatter_xy_step_'
    approx = 'beams_linear_xy_step_'
    approx_data = 'beams_linear_data_step_'
    legend = 'beams_legend_step_'
    original_values_a = 'beams_original_values_a_step_'
    original_values_b = 'beams_original_values_b_step_'


class CfgDataset():
    images_file_path = 'wc_co_images'
    preprocess_images_file_path = 'wc_co_images_preprocess'
    images_urls = ['https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co8.jpg',
                   'https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co11.jpg',
                   'https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co6_2.jpg',
                   'https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co15.jpg',
                   'https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co25.jpg']

    images_crop_urls = ['https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co8-crop.jpg',
                        'https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co11-crop.jpg',
                        'https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co6_2-crop.jpg',
                        'https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co15-crop.jpg',
                        'https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co25-crop.jpg']

    images_crop_preproc_urls = ['https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co8-crop-preproc.jpg',
                                'https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co11-crop-preproc.jpg',
                                'https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co6_2-crop-preproc.jpg',
                                'https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co15-crop-preproc.jpg',
                                'https://pobedit.s3.us-east-2.amazonaws.com/default_images/Ultra_Co25-crop-preproc.jpg']

    images_names = np.array(
        [['Ultra_Co8.jpg'], ['Ultra_Co11.jpg'], ['Ultra_Co6_2.jpg'], ['Ultra_Co15.jpg'], ['Ultra_Co25.jpg']])

    types = ['средние зерна', 'мелкие зерна', 'мелкие зерна', 'крупные зерна', 'средне-мелкие зерна']
