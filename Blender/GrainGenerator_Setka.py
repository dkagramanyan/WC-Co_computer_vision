import bpy
from random import random
from mathutils import Vector, Euler
import math
import numpy as np


# единицы измерения - метры

# функция, которая по координатам ищет номер ячейки, в которой находится кубик
def findGridLocation(loc, size, n_xy):
    gridNum = int(loc[0] // (2 * size)) + n_xy * int(loc[1] // (2 * size)) + (n_xy ** 2) * int(loc[2] // (2 * size))
    print(loc, gridNum)
    return gridNum


# Проверяет, пересекаются ли кубы
# Закомменченная часть - исправленный способ, который предложила Лиза
# Куб сначала проверяет расстояние между центрами, потом берёт один куб, домножает координаты его вершин на матрицу
# поворота, потом умножает на матрицу поворота другого кубика и определяет по проекциям, пересекаются ли кубы
'''
def ifIntersept(loc1, loc2, rot1, rot2, size):

    #checks the destination between 2 cubes
    if (loc2 - loc1).length > (size*(3**(1/2))):
        return True

    else:
        #creates a list of coordinates of every vertice
        coord_list_1 = []
        coord_list_2 = []

        #calculates the coordinates and adds to list
        coord_list_1.append([loc1[0]+size/2, loc1[1]+size/2, loc1[2]+size/2])
        coord_list_1.append([loc1[0]+size/2, loc1[1]+size/2, loc1[2]-size/2])
        coord_list_1.append([loc1[0]+size/2, loc1[1]-size/2, loc1[2]+size/2])
        coord_list_1.append([loc1[0]+size/2, loc1[1]-size/2, loc1[2]-size/2])
        coord_list_1.append([loc1[0]-size/2, loc1[1]+size/2, loc1[2]+size/2])
        coord_list_1.append([loc1[0]-size/2, loc1[1]+size/2, loc1[2]-size/2])
        coord_list_1.append([loc1[0]-size/2, loc1[1]-size/2, loc1[2]+size/2])
        coord_list_1.append([loc1[0]-size/2, loc1[1]-size/2, loc1[2]-size/2])

        coord_list_2.append([loc2[0]+size/2, loc2[1]+size/2, loc2[2]+size/2])
        coord_list_2.append([loc2[0]+size/2, loc2[1]+size/2, loc2[2]-size/2])
        coord_list_2.append([loc2[0]+size/2, loc2[1]-size/2, loc2[2]+size/2])
        coord_list_2.append([loc2[0]+size/2, loc2[1]-size/2, loc2[2]-size/2])
        coord_list_2.append([loc2[0]-size/2, loc2[1]+size/2, loc2[2]+size/2])
        coord_list_2.append([loc2[0]-size/2, loc2[1]+size/2, loc2[2]-size/2])
        coord_list_2.append([loc2[0]-size/2, loc2[1]-size/2, loc2[2]+size/2])
        coord_list_2.append([loc2[0]-size/2, loc2[1]-size/2, loc2[2]-size/2])

        for coord in coord_list_2:
            coord[1]=coord[1]*math.cos(rot2[0]) - coord[2]*math.sin(rot2[0])
            coord[2]=coord[1]*math.sin(rot2[0]) + coord[2]*math.cos(rot2[0])

            coord[0]=coord[0]*math.cos(rot2[1]) + coord[2]*math.sin(rot2[1])
            coord[2]=-coord[0]*math.sin(rot2[1]) + coord[2]*math.cos(rot2[1])

            coord[0]=coord[0]*math.cos(rot2[2]) - coord[1]*math.sin(rot2[2])
            coord[1]=coord[0]*math.sin(rot2[2]) + coord[1]*math.cos(rot2[2])

        for coord in coord_list_2:
            coord[1]=coord[1]*math.cos(rot1[0]) + coord[2]*math.sin(rot1[0])
            coord[2]=-coord[1]*math.sin(rot1[0]) + coord[2]*math.cos(rot1[0])

            coord[0]=coord[0]*math.cos(rot1[1]) - coord[2]*math.sin(rot1[1])
            coord[2]=coord[0]*math.sin(rot1[1]) + coord[2]*math.cos(rot1[1])

            coord[0]=coord[0]*math.cos(rot1[2]) + coord[1]*math.sin(rot1[2])
            coord[1]=-coord[0]*math.sin(rot1[2]) + coord[1]*math.cos(rot1[2])

        #Create a list of x, y, z coordinates of each cube
        x1_list = []
        y1_list = []
        z1_list = []

        for c in coord_list_1:
            x1_list.append(c[0])
            y1_list.append(c[1])
            z1_list.append(c[2])

        x2_list = []
        y2_list = []
        z2_list = []

        for c in coord_list_2:
            x2_list.append(c[0])
            y2_list.append(c[1])
            z2_list.append(c[2])

        #checks if cubes intersept        
        if ( (max(x1_list) >= min(x2_list)) and (min(x1_list) <= max(x2_list))
            and (max(y1_list) >= min(y2_list)) and (min(y1_list) <= max(y2_list))
            and (max(z1_list) >= min(z2_list)) and (min(z1_list)) <= max(z2_list) ):
                return True
        else:
            return False
'''


# Почти то же самое, но здесь каждый куб умножается на свою матрицу поворота и там уже смотрятся
# С редкими ошибками, но работает, просто я пытался понять, в функции ли дело

def ifIntersept(loc1, loc2, rot1, rot2, size):
    # проверяет расстояние между двумя кубами
    if (loc2 - loc1).length > (size * (3 ** (1 / 2))):
        return True
    else:

        # список координат всех вершин каждого куба
        coord_list_1 = []
        coord_list_2 = []

        # считает координаты(без учёта поворота)
        coord_list_1.append([loc1[0] + size / 2, loc1[1] + size / 2, loc1[2] + size / 2])
        coord_list_1.append([loc1[0] + size / 2, loc1[1] + size / 2, loc1[2] - size / 2])
        coord_list_1.append([loc1[0] + size / 2, loc1[1] - size / 2, loc1[2] + size / 2])
        coord_list_1.append([loc1[0] + size / 2, loc1[1] - size / 2, loc1[2] - size / 2])
        coord_list_1.append([loc1[0] - size / 2, loc1[1] + size / 2, loc1[2] + size / 2])
        coord_list_1.append([loc1[0] - size / 2, loc1[1] + size / 2, loc1[2] - size / 2])
        coord_list_1.append([loc1[0] - size / 2, loc1[1] - size / 2, loc1[2] + size / 2])
        coord_list_1.append([loc1[0] - size / 2, loc1[1] - size / 2, loc1[2] - size / 2])

        coord_list_2.append([loc2[0] + size / 2, loc2[1] + size / 2, loc2[2] + size / 2])
        coord_list_2.append([loc2[0] + size / 2, loc2[1] + size / 2, loc2[2] - size / 2])
        coord_list_2.append([loc2[0] + size / 2, loc2[1] - size / 2, loc2[2] + size / 2])
        coord_list_2.append([loc2[0] + size / 2, loc2[1] - size / 2, loc2[2] - size / 2])
        coord_list_2.append([loc2[0] - size / 2, loc2[1] + size / 2, loc2[2] + size / 2])
        coord_list_2.append([loc2[0] - size / 2, loc2[1] + size / 2, loc2[2] - size / 2])
        coord_list_2.append([loc2[0] - size / 2, loc2[1] - size / 2, loc2[2] + size / 2])
        coord_list_2.append([loc2[0] - size / 2, loc2[1] - size / 2, loc2[2] - size / 2])

        # Домножаем на матрицы поворота
        for coord in coord_list_1:
            coord[1] = coord[1] * math.cos(rot1[0]) + coord[2] * math.sin(rot1[0])
            coord[2] = -coord[1] * math.sin(rot1[0]) + coord[2] * math.cos(rot1[0])

            coord[0] = coord[0] * math.cos(rot1[1]) + coord[2] * math.sin(rot1[1])
            coord[2] = -coord[0] * math.sin(rot1[1]) + coord[2] * math.cos(rot1[1])

            coord[0] = coord[0] * math.cos(rot1[2]) - coord[1] * math.sin(rot1[2])
            coord[1] = -coord[0] * math.sin(rot1[2]) + coord[1] * math.cos(rot1[2])

        for coord in coord_list_2:
            coord[1] = coord[1] * math.cos(rot2[0]) + coord[2] * math.sin(rot2[0])
            coord[2] = -coord[1] * math.sin(rot2[0]) + coord[2] * math.cos(rot2[0])

            coord[0] = coord[0] * math.cos(rot2[1]) + coord[2] * math.sin(rot2[1])
            coord[2] = -coord[0] * math.sin(rot2[1]) + coord[2] * math.cos(rot2[1])

            coord[0] = coord[0] * math.cos(rot2[2]) - coord[1] * math.sin(rot2[2])
            coord[1] = -coord[0] * math.sin(rot2[2]) + coord[1] * math.cos(rot2[2])

        # Кидает в кучу координаты по x, y и z для каждого куба
        x1_list = []
        y1_list = []
        z1_list = []

        for c in coord_list_1:
            x1_list.append(c[0])
            y1_list.append(c[1])
            z1_list.append(c[2])

        x2_list = []
        y2_list = []
        z2_list = []

        for c in coord_list_2:
            x2_list.append(c[0])
            y2_list.append(c[1])
            z2_list.append(c[2])

        # проверяет на пересечение
        if ((max(x1_list) >= min(x2_list)) and (min(x1_list) <= max(x2_list))
                and (max(y1_list) >= min(y2_list)) and (min(y1_list) <= max(y2_list))
                and (max(z1_list) >= min(z2_list)) and (min(z1_list)) <= max(z2_list)):
            return True
        else:
            return False


# Приколы с камерой, на алгоритм никак не влияет
cam = bpy.data.cameras["Camera"]
cam.lens = 67
cam.sensor_width = 10
cam.display_size = 0.1

# размеры зёрен обращцов
'''
#для генерации на харизме
grain_sizes = [0.005, 0.003, 0.001]
'''

grain_sizes = [0.05]  # для генерации локально

sample_number = 0

for cubeRadius in grain_sizes:

    sample_number += 1

    # min and max values for each axis for the random numbers
    # считает на основе размера кубика, чтобы оставить "клетку"
    ranges = {
        'x': {'min': 2 * cubeRadius, 'max': 2 * cubeRadius + 1},
        'y': {'min': 2 * cubeRadius, 'max': 2 * cubeRadius + 1},
        'z': {'min': 2 * cubeRadius, 'max': 2 * cubeRadius + 0.05}
    }

    size = (0.05 / cubeRadius ** 3) * (3 / 7)  # количество кубов

    # ищет количество ячеек по каждой оси (n_xy и n_z)
    if 1 % (2 * cubeRadius) == 0:
        n_xy = int(1 / (2 * cubeRadius)) + 2
        n_z = int(0.05 / (2 * cubeRadius)) + 2
    else:
        n_xy = int(1 // (2 * cubeRadius)) + 3
        n_z = int(0.05 // (2 * cubeRadius)) + 3

    div_number = (n_xy ** 2) * n_z  # общее количество ячеек

    cubes = []  # список координат
    rotations = []  # список углов вращения

    # добавляет списки в списки, чтобы потом быстро искать нужные кубики просто по номеру ячейки
    for i in range(div_number):
        cubes.append([])
        rotations.append([])

    # Generates a random number within the axis minmax range
    randLocInRange = lambda axis: ranges[axis]['min'] + random() * (ranges[axis]['max'] - ranges[axis]['min'])

    # Generate a random 3D coordinate
    loc = Vector([randLocInRange(axis) for axis in ranges.keys()])
    # Add coordinate to cube list
    cubes[findGridLocation(loc, cubeRadius, n_xy)].append(loc)

    # Generates a random rotation
    rot = Euler((random(), random(), random()), 'XYZ')
    # Add rotation to rotations list
    rotations[findGridLocation(loc, cubeRadius, n_xy)].append(rot)

    # Add the first cube (others will be duplicated from it)
    # Note: in blender 2.8 size arg is used instead of radius
    bpy.ops.mesh.primitive_cube_add(size=cubeRadius, location=loc, rotation=rot)

    cube = bpy.context.scene.objects['Cube']
    mat = bpy.data.materials.get("Material")
    cube.data.materials.append(mat)

    # Add all other cubes
    c = 0
    while c <= size:
        while True:
            flag = True
            # Generate a random 3D coordinate
            loc = Vector([randLocInRange(axis) for axis in ranges.keys()])
            # Generate a random rotation
            rot = Euler((random(), random(), random()), 'XYZ')

            gridNum = findGridLocation(loc, cubeRadius, n_xy)  # ищет номер ячейки
            nearGrids = []  # список ячеек, которые надо проверить
            nearGrids.append(gridNum + 1)
            nearGrids.append(gridNum - 1)

            nearGrids.append(gridNum + n_xy)
            nearGrids.append(gridNum - n_xy)
            nearGrids.append(gridNum + 1 + n_xy)
            nearGrids.append(gridNum - 1 + n_xy)
            nearGrids.append(gridNum + 1 - n_xy)
            nearGrids.append(gridNum - 1 - n_xy)

            nearGrids.append(gridNum + n_xy + n_xy ** 2)
            nearGrids.append(gridNum - n_xy - n_xy ** 2)

            nearGrids.append(gridNum + n_xy ** 2)
            nearGrids.append(gridNum + 1 + n_xy ** 2)
            nearGrids.append(gridNum - 1 + n_xy ** 2)
            nearGrids.append(gridNum + n_xy + n_xy ** 2)
            nearGrids.append(gridNum - n_xy + n_xy ** 2)
            nearGrids.append(gridNum + 1 + n_xy + n_xy ** 2)
            nearGrids.append(gridNum - 1 + n_xy + n_xy ** 2)
            nearGrids.append(gridNum + 1 - n_xy + n_xy ** 2)
            nearGrids.append(gridNum - 1 - n_xy + n_xy ** 2)

            nearGrids.append(gridNum - n_xy ** 2)
            nearGrids.append(gridNum + 1 - n_xy ** 2)
            nearGrids.append(gridNum - 1 - n_xy ** 2)
            nearGrids.append(gridNum + n_xy - n_xy ** 2)
            nearGrids.append(gridNum - n_xy - n_xy ** 2)
            nearGrids.append(gridNum + 1 + n_xy - n_xy ** 2)
            nearGrids.append(gridNum - 1 + n_xy - n_xy ** 2)
            nearGrids.append(gridNum + 1 - n_xy - n_xy ** 2)
            nearGrids.append(gridNum - 1 - n_xy - n_xy ** 2)

            # проверяет каждую ячейку
            # если хотя бы с одним кубиком пересекается- генерирует новую координату
            for grid in nearGrids:
                if len(cubes[grid]) != 0:
                    for l in range(len(cubes[grid])):
                        if ifIntersept(loc, cubes[grid][l], rot, rotations[grid][l], cubeRadius) is False:
                            flag = False
            if flag is True:
                break

        # Add coordinate to cube list
        cubes[gridNum].append(loc)
        rotations[gridNum].append(rot)
        dupliCube = cube.copy()
        dupliCube.location = loc
        dupliCube.rotation_euler = rot

        # bpy.context.scene.objects.link( dupliCube )
        # in blender 2.8 an api change requires to use the collection instead of the scene
        bpy.context.collection.objects.link(dupliCube)
        c += 1

    # здесь и далее - шаманство с камерой и удалением кубиков после получения снимков
    # чтобы генерировать следующий образец
    # Creates a list of camera locations
    '''
    #для генерации на харизме
    camera_locations = []
    y = 95+2*cubeRadius
    while y > 2*cubeRadius:
        x = 2*cubeRadius
        while x < 95+2*cubeRadius:
            camera_locations.append((x/100.0, y/100.0, 0.95))
            x+=10
        y-=10
    '''

    camera_locations = [(2 * cubeRadius, 0.95 + 2 * cubeRadius, 0.95),
                        (0.1 + 2 * cubeRadius, 0.95 + 2 * cubeRadius, 0.95),
                        (0.2 + 2 * cubeRadius, 0.95 + 2 * cubeRadius, 0.95)]  # для генерации локально

    # makes 100 images
    photo_number = 0
    for cam_location in camera_locations:
        photo_number += 1

        # change the location of camera
        camobj = bpy.data.objects["_cam"]
        camobj.location = cam_location

        # render and save the image
        scene = bpy.context.scene
        scene.camera = camobj
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = 'C:/Users/HOME/Desktop/Pobedit/GrainGenerator/blender' + str(sample_number) + '_' + str(
            photo_number) + '.png'
        bpy.ops.render.render(write_still=1)

    '''
    #delete all the grains
    bpy.ops.object.select_all(action='SELECT')
    bpy.data.objects['_cam'].select_set(False)
    bpy.ops.object.delete()
    '''
