import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

from lmfit.models import Model

from sklearn.cluster import KMeans

from shapely.geometry import Polygon

from radio_beam.commonbeam import getMinVolEllipse

from scipy import ndimage as ndi
from scipy.spatial import distance

from skimage import io
from skimage.measure import EllipseModel
from skimage.color import rgb2gray
from skimage import filters
from skimage.morphology import disk
from skimage.measure import approximate_polygon


from PIL import Image, ImageDraw, ImageFilter,ImageOps

import copy
import cv2

from scipy.spatial import ConvexHull

class grainPreprocess(): 

    @classmethod
    def imdivide(cls,image,h,side):
        #
        # возвращает левую или правую часть полученного изображения
        #
        width,height = image.size
        sides={'left':0,'right':1}
        shape=[(0,0,width//2,height-h),(width//2,0,width,height-h)]
        return image.crop(shape[sides[side]])
    
    @classmethod
    def combine(cls,image,h,k=0.5): 
        #
        #  накладывает левую и правые части изображения
        #  если k=1, то на выходе будет левая часть изображения, если k=0, то будет правая часть
        #
        left_img=cls.imdivide(image,h,'left')
        right_img=cls.imdivide(image,h,'right')

        l=k
        r=1-l
        gray=np.array(left_img)*l
        gray+=np.array(right_img)*r
        gray=gray.astype('uint8')
        img=rgb2gray(gray)
        return img

    @classmethod
    def do_otsu(cls,img):
        #
        # бинаризация отсу
        #
        global_thresh=filters.threshold_otsu(img)
        binary_global = img > global_thresh

        return binary_global
    
    
    @classmethod
    def image_preprocess(cls,image,h=135,k=1):
        #
        # комбинация медианного фильтра, биноризации и гражиента
        # у зерен значение пикселя - 0, у регионов связ. в-ва - 1,а у их границы - 2
        #
        combined=cls.combine(image,h,k)
        denoised = filters.rank.median(combined, disk(3))
        binary=cls.do_otsu(denoised).astype('uint8')
        grad = abs(filters.rank.gradient(binary, disk(1))).astype('uint8')
        bin_grad=1-binary+grad
        new_image=(bin_grad>0).astype('uint8')*255

        return new_image
    
    @classmethod
    def image_preprocess_kmeans(cls,image,h=135,k=1,n_clusters=3,pos=1):
        #
        # выделение границ при помощи кластеризации 
        # и выравнивание шума медианным фильтром
        # pos отвечает за выбор кластера, который будет отображен на возвращенном изображении
        #
        combined = cls.combine(image,h,k)
        
        clustered,colors = grainMorphology.kmeans_image(combined,n_clusters)
        cluster = clustered==colors[pos]
        cluster = np.array(cluster*255,dtype='uint8')
        
        new_image = filters.median(cluster,disk(2))
        return new_image
    
class grainMorphology():
    
    @classmethod
    def kmeans_image(cls,image,n_clusters=3):
        #
        # кластеризует при помощи kmeans
        # и возвращает изображение с нанесенными цветами кластеров
        #
        img=image.copy()

        size = img.shape
        img = img.reshape(-1, 1)

        model = KMeans(n_clusters=n_clusters)
        clusters = model.fit_predict(img)

        colors=[]
        for i in range(n_clusters):
            color=np.median(img[clusters == i]) # медианное значение пикселей у кластера
            img[clusters == i] = color
            colors.append(int(color))

        img = img.reshape(size)
        colors.sort()

        return img,colors

class grainFig():
    
    @classmethod
    def line(cls,point1,point2):
        #
        # возвращает растровые координаты прямой между двумя точками 
        #
        line=[]

        x1,y1=point1[0],point1[1]
        x2,y2=point2[0],point2[1]

        dx = x2 - x1
        dy = y2 - y1

        sign_x = 1 if dx>0 else -1 if dx<0 else 0
        sign_y = 1 if dy>0 else -1 if dy<0 else 0

        if dx < 0: dx = -dx
        if dy < 0: dy = -dy

        if dx > dy:
            pdx, pdy = sign_x, 0
            es, el = dy, dx
        else:
            pdx, pdy = 0, sign_y
            es, el = dx, dy

        x, y = x1, y1
        error, t = el/2, 0        

        line.append((x, y))

        while t < el:
            error -= es
            if error < 0:
                error += el
                x += sign_x
                y += sign_y
            else:
                x += pdx
                y += pdy
            t += 1
            line.append((x, y))
        return np.array(line).astype('int')
    
    @classmethod
    def rect(cls,point1,point2,r):
        #
        # возвращает растровые координаты прямоугольника ширины 2r,
        # построеного между двумя точками 
        #
        x1,y1=point1[0],point1[1]
        x2,y2=point2[0],point2[1]

        l1,l2=(x2-x1),(y2-y1)

        l_len=(l1**2+l2**2)**0.5
        l_len=int(l_len)

        a=(x1-r*l2/l_len),(y1+r*l1/l_len)
        b=(x1+r*l2/l_len),(y1-r*l1/l_len)

        side=cls.line(a,b)

        # a -> c
        lines=np.zeros((side.shape[0],l_len*2,2),dtype='int64')

        for i,left_point in enumerate(side):
            right_point=(left_point[0]+l1),(left_point[1]+l2)
            line_points=cls.get_line(left_point,right_point)
            for j,point in enumerate(line_points):
                lines[i,j]=point

        return lines
    
    
class grainMark():
    @classmethod
    def mark_corners_and_classes(cls,image,max_num=100000,sens=0.1,max_dist=1):

        corners = cv2.goodFeaturesToTrack(image,max_num,sens, max_dist)
        corners = np.int0(corners)
        x=copy.copy(corners[:,0,1])
        y=copy.copy(corners[:,0,0])
        corners[:,0,0],corners[:,0,1]=x,y

        classes = filters.rank.gradient(image, disk(1)) < 250
        classes,num = ndi.label(classes)
        return corners,classes,num
    
    @classmethod
    def get_nodes_corners_classes_classesNum(cls,
                                             image,
                                             max_corners_per_class,
                                             max_num=100000,
                                             sens=0.1,
                                             max_dist=1):
        corners,classes,classes_num=cls.mark_corners_and_classes(image,max_num,sens, max_dist)
        shape=(classes_num+1,max_corners_per_class)
        nodes=np.zeros(shape,dtype='int64')
        r=2
        for i,corner in enumerate(corners):
            x=corner[0][0]
            y=corner[0][1]
            cls=classes[x,y]
            real_cls=0
            flag=True
            x_lin=range(x-r,x+r)
            y_lin=range(y-r,y+r)
            while flag:
                for xi in range(x-r,x+r):
                    for yi in y_lin:
                        if classes[xi,yi]!=0 and classes[xi,yi]!=1 and classes[xi,yi]!=2:
                            real_cls=classes[xi,yi]
                            flag=False
                        elif yi==y_lin[-1] and xi==x_lin[-1]:
                            flag=False


            index=np.where(nodes[real_cls]==0)[0][0]
            nodes[real_cls,index]=i
            nodes[real_cls,-1]+=1 

        return nodes,classes,classes_num,corners


   
    
    @classmethod
    def mean_pixel(cls,image,point1,point2,r):


        val1,num1=cls.draw_rect(image,point1,point2,r)
        val2,num2=cls.draw_rect(image,point2,point1,r)
        val=val1+val2
        num=num1+num2

        if num!=0 and val!=0:
            mean=(val/num)/255
            dist=distance.euclidean(point1,point2)
        else:
            mean=1
            dist=1
        return mean,dist
    
    @classmethod
    def estimate_edges(cls,image,node,corners,position=0,radius=2):
        # вычисление расстояния и среднего значения между 
        # точкой с индексом posion и остальными точками пустоты
        v1=node[position]
        vals=np.zeros((len(node),2))
        if v1!=0:
            for i,v2 in enumerate(node):

                if v2!=0 and v1!=v2  :
                    vals[i,0]=v2
                    point1=corners[v1]
                    point2=corners[v2]
                    mean,dist=cls.mean_pixel(image,point1[0],point2[0],radius)
                    vals[i,1]=abs(mean-0.5)*dist
               #     vals[i,2]=dist

      #  vals=vals[np.argsort(vals[:,2])]
        return vals
    
    @classmethod
    def sort_perimetr(cls,orig_image,nodes,corners,eps,radius):
        
        img=Image.fromarray(orig_image)
        img = ImageOps.expand(img,border=radius,fill='black')
        image=np.array(img)

        new_nodes=np.zeros(nodes.shape,dtype='int64')

        # проходим по каждой пустоте
        for i,orig_node in enumerate(nodes):
        # node - пустота
            v1_index=0
            v1=orig_node[v1_index] # фиксируем первый угол и затем будем считать относительно него
            step=1
            new_nodes[i,0]=v1
            flag=True
            orig_node_len=orig_node[-1]
            node=list(copy.copy(orig_node[:orig_node_len]))
           # norm_val=mean_dist_points(node,corners)
          #  print('i=',i, ' max=',nodes.shape[0])
          #  print('-------------')
           # print('old_node',node)
            #print('new_node')
            if v1!=0 and len(node)>1:
                while flag:
            #        print('delete_node,iteration start',node)
             #       print('v1',v1)
                    vals=cls.estimate_edges(image,node,corners,position=v1_index,radius=radius)
                    non_zero_indeces=np.where((vals[:,0][:orig_node_len]!=0) )[0]
                    vals=vals[non_zero_indeces]
                #    print('new kernel point')

                    min_val=vals.min(axis=0)[1]
           #         for j,val in enumerate(vals):
         #               mean=val[1]
          #              if abs(mean-0.5)<=eps:
            #                v2=val[0]
            #                break
            #            elif j==vals.shape[0]-1:
              #              v2=val[0]
             #               break

                  #  print('min_val',min_val)
                    v2_vals_index=np.where(vals[:,1]==min_val )[0][0]
                    v2=vals[v2_vals_index][0]
               #     print('v2',v2)

                    new_nodes[i,step]=v2
                    node.pop(v1_index)

                    v1=v2
                    v2_index=node.index(v2)

                #    print('v2_index',v2_index)
                    v1_index=v2_index


               #     if len(node)==1:
              #           new_nodes[i,step+1]=node[0]
             #           flag=False

                    step+=1
                    if step==orig_node_len:
                        flag=False

            else:
                continue
            new_nodes[i,-1]= orig_node_len
        return new_nodes
    
    @classmethod
    def sor_perimetr_hull(cls,image,nodes,corners,color=(51,51,51)):
        new_nodes=np.zeros(nodes.shape,dtype='int64')


        for i,node in enumerate(nodes):

            if node[0]!=0 and node[-1]>2:
                l=node[-1]
                points=np.zeros((l,2))
                for j,point in enumerate(node[:l]):
                    points[j]=corners[point,0] 

                hull=ConvexHull(points)
                for simplex in hull.simplices:
                    (x1,x2)=points[simplex, 0]
                    (y1,y2)= points[simplex, 1]


        for j,node in enumerate(nodes):
            if len(node)>1:
                r=3
                len_node=node[-1]

                for i,point in enumerate(node[: len_node]):
                    point2=corners[point][0]
                    x2,y2=point2[0],point2[1]

            else:
                continue



        return  img
        
       
    
    @classmethod
    def get_row_contours(cls,image):
        #
        # возвращает набор точек контуров 
        #
        edges = cv2.Canny(image,0,255,L2gradient=False)

        # направление обхода контура по часовой стрелке
        contours,_ = cv2.findContours( edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        contours=np.array(contours)
        for i,cnt in enumerate(contours):
            contours[i]=cnt[:,0]
        return contours
    
    @classmethod
    def get_contours(cls,image):
        #
        # уменьшение количества точек контура при помощи алгоритма Дугласа-Пекера
        #
        contours=cls.get_row_contours(image)

        new_contours=[]
        for j,cnt in enumerate(contours):
            if len(cnt)>1:
                coords = approximate_polygon(cnt, tolerance=3)
                new_contours.append(coords)
            else:
                continue

        return new_contours
        
    @classmethod
    def get_angles(cls,image):
        #
        # считаем углы с направлением обхода контура против часовой стрелки, углы >180 градусов учитываются
        #
        approx=cls.get_contours(image)

        # вычисление угла
        angles=[]
        for k,cnt in enumerate(approx):
            l=len(cnt)
            if l>2:
                for i in range(l)[1:l-1]:
                    point1=cnt[i-1]
                    point2=cnt[i]
                    point3=cnt[i+1]
                    x1,y1=point1[1],point1[0]
                    x2,y2=point2[1],point2[0]
                    x3,y3=point3[1],point3[0]

                    v1=np.array((x1-x2,y1-y2)).reshape(1,2)
                    v2=np.array((x3-x2,y3-y2)).reshape(1,2)

                    dot=np.dot(v1[0],v2[0])
                    dist1 = np.linalg.norm(v1[0])
                    dist2 = np.linalg.norm(v2[0])
                    cos=dot/(dist1*dist2)

                    v=np.concatenate([v1,v2])
                    det=np.linalg.det(v)

                    if abs(cos)<1:
                        ang=int(np.arccos(cos)*180/np.pi) 
                        if det<0 :
                            angles.append(ang)
                        else:
                            angles.append(360-ang)

        return np.array(angles)
    
    @classmethod
    def get_mvee_params(cls,image,tol=0.2):
        #
        # возвращает полуоси и угол поворота фигуры minimal volume enclosing ellipsoid,
        # которая ограничивает исходные точки контура эллипсом
        # 
        approx=grainMark.get_row_contours(image)
        a_beams=[]
        b_beams=[]
        angles=[]
        centroids=[]
        for i,cnt in enumerate(approx):
            if len(cnt)>2:
                cnt=np.array(cnt)
                polygon=Polygon(cnt)

                x_centroid,y_centroid=polygon.centroid.coords[0]
                points=cnt-(x_centroid,y_centroid)

                x_norm,y_norm=points.mean(axis=0)
                points = (points-(x_norm,y_norm))

                data=getMinVolEllipse(points,tol)

                data=np.array(data)
                xc,yc=data[0][0]
                a,b=data[1]
                sin=data[2][0,1]
                angle=-np.arcsin(sin)

                a_beams.append(a)
                b_beams.append(b)
                angles.append(angle)
                centroids.append([x_centroid+x_norm,y_centroid+y_norm])

        a_beams=np.array(a_beams,dtype='int32')
        b_beams=np.array(b_beams,dtype='int32')
        angles=np.array(angles,dtype='float32')
        centroids=np.array(centroids,dtype='int32')


        return a_beams,b_beams,angles,centroids



class grainShow():
    
    @classmethod
    def img_show(cls,image,N=20,cmap=plt.cm.nipy_spectral):
        #
        # выводит изображение image
        #

        plt.figure(figsize=(N,N))
        plt.axis('off')
        plt.imshow(image,cmap=cmap)
        plt.show()
    
    @classmethod
    def corners_classes(cls,nodes,classes,max_num=2000,size=5,show=False): 
        node_corner_numbers=np.zeros((classes.shape[0],1))
        for i,node in enumerate(nodes):
            for corner in node:
                if corner:
                    node_corner_numbers[i]+=1

        std=node_corner_numbers.std(axis=0)[0]
        mean=node_corner_numbers.mean(axis=0)[0]

        print('std: ',round(std,3))
        print('mean: ',round(mean,3))
        print('количесвто углов: ',classes.shape[0])
        if show:
            fig, ax = plt.subplots(figsize=(size,size))
            ax.set_ylabel('Количество углов у пустоты',size=20)
            ax.set_xlabel('Порядковый номер пустоты',size=20)
            plt.plot(node_corner_numbers[:max_num])
        return node_corner_numbers
    
    @classmethod
    def classes_corners(cls,nodes,corners,max_num=2000,size=5,show=False):   
        corner_distr=np.zeros((corners.shape[0]))

        for i in range(corners.shape[0]):
            for node in nodes:
                node_len=node[-1]
                if i in node[:node_len]:
                    corner_distr[i]+=1

        std=corner_distr.std(axis=0)
        mean=corner_distr.mean(axis=0)

        print('std: ',round(std,3))
        print('mean: ',round(mean,3))
        print('количество пустот: ',nodes.shape[0])
        if show:
            fig, ax = plt.subplots(figsize=(size,size))
            ax.set_ylabel('Количество вхождений в пустоту',size=20)
            ax.set_xlabel('Порядковый номер угла',size=20)
            plt.plot(corner_distr[:max_num])
            plt.show()
        return corner_distr

class grainDraw():
    
    @classmethod
    def draw_corners(cls,image,corners,color=255):
        image=copy.copy(image)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(image, (x, y), 3, color, -1)

        return image
    
    @classmethod
    def draw_edges_nodes(cls,image,nodes,corners,color=(51,51,51)):
        new_image=copy.copy(image)
        im = Image.fromarray(np.uint8(cm.gist_earth(new_image)*255))
        draw = ImageDraw.Draw(im)
        for j,node in enumerate(nodes):
            if len(node)>1:
             #   print('i=',j)
            #    print(node)
                point1=corners[node[0]][0]
                x1,y1=point1[0],point1[1]

                x_start,y_start=point1[0],point1[1]
                r1=5
                r=3
                draw.ellipse((y1-r1,x1-r1,y1+r1,x1+r1), fill=color, width=10)
                len_node=node[-1]
           #     print(node[:len_node])
                for i,point in enumerate(node[: len_node]):
                    point2=corners[point][0]
                    x2,y2=point2[0],point2[1]

                    draw.ellipse((y2-r,x2-r,y2+r,x2+r), fill=color, width=4)
                    draw.line((y1,x1,y2,x2), fill=color, width=4)
                    x1,y1=x2,y2

                draw.line((y_start,x_start,y1,x1), fill=(100,100,100), width=4)
            else:
                continue

        img=np.array(im)

        return  img
    
    @classmethod
    def draw_edges_hull(cls,image,nodes,corners,color=(51,51,51)):
        new_image=copy.copy(image)
        im = Image.fromarray(np.uint8(cm.gist_earth(new_image)*255))
        draw = ImageDraw.Draw(im)

        for i,node in enumerate(nodes):

            if node[0]!=0 and node[-1]>2:
                l=node[-1]
                points=np.zeros((l,2))
                for j,point in enumerate(node[:l]):
                    points[j]=corners[point,0] 

                hull=ConvexHull(points)
                for simplex in hull.simplices:
                    (x1,x2)=points[simplex, 0]
                    (y1,y2)= points[simplex, 1]

                    points[simplex, 0], points[simplex, 1]
                    draw.line((y1,x1,y2,x2), fill=(100,100,100), width=4)
        for j,node in enumerate(nodes):
            if len(node)>1:
                r=3
                len_node=node[-1]

                for i,point in enumerate(node[: len_node]):
                    point2=corners[point][0]
                    x2,y2=point2[0],point2[1]
                    draw.ellipse((y2-r,x2-r,y2+r,x2+r), fill=color, width=4)
            else:
                continue

        img=np.array(im)

        return  img
    
    @classmethod
    def draw_edges(cls,image,cnts,color=(50,50,50)):
        #
        # рисует на изображении линии по точкам контура cnts
        #
        new_image=copy.copy(image)
        im = Image.fromarray(np.uint8(cm.gist_earth(new_image)*255))
        draw = ImageDraw.Draw(im)

        for j,cnt in enumerate(cnts):
            if len(cnt)>1:
                point=cnt[0]
                x1,y1=point[1],point[0]
                r=4

                for i,point2 in enumerate(cnt):
                    p2=point2

                    x2,y2=p2[1],p2[0]

                    draw.ellipse((y2-r,x2-r,y2+r,x2+r), fill=color, width=5)
                    draw.line((y1,x1,y2,x2), fill=(100,100,100), width=4)
                    x1,y1=x2,y2

            else:
                continue

        img=np.array(im)

        return  img
        
class grainStats():
    
    @classmethod
    def stats_preprocess(cls,array,step):
        #
        # приведение углов к кратости, например 0,step,2*step и тд
        #
        array_copy=array.copy()

        for i,a in enumerate(array_copy):
            while array_copy[i]%step!=0:
                array_copy[i]+=1

        array_copy_set=np.sort(np.array(list(set(array_copy))))
        dens_curve=[]
        for arr in array_copy_set:
            num=0
            for ar in array_copy:
                if arr==ar:
                    num+=1
            dens_curve.append(num)
        return np.array(array_copy),array_copy_set,np.array(dens_curve)
    
    @classmethod
    def gaussian(cls,x, mu, sigma,amp=1):
        #
        # возвращает нормальную фунцию по заданным параметрам
        #
        return np.array((amp/(np.sqrt(2*np.pi)*sigma))*np.exp(-(x-mu)**2 / (2*sigma**2)))
    
    @classmethod
    def gaussian_bimodal(cls,x,mu1,mu2,sigma1,sigma2,amp1=1,amp2=1):
        #
        # возвращает бимодальную нормальную фунцию по заданным параметрам
        #
        return cls.gaussian(x,mu1,sigma1,amp1)+cls.gaussian(x,mu2,sigma2,amp2)
    
    @classmethod
    def gaussian_termodal(cls,x,mu1,mu2,mu3,sigma1,sigma2,sigma3,amp1=1,amp2=1,amp3=1):
        #
        # возвращает термодальную нормальную фунцию по заданным параметрам
        #
        return cls.gaussian(x,mu1,sigma1,amp1)+cls.gaussian(x,mu2,sigma2,amp2)+cls.gaussian(x,mu3,sigma3,amp3)
    
    @classmethod
    def ellipse(cls,a,b,angle,xc=0,yc=0,num=50):
        #
        #  возвращает координаты эллипса, построенного по заданным параметрам
        #  по умолчанию центр (0,0)
        #  угол в радианах, уменьшение угла обозначает поворот эллипса по часовой стрелке
        #
        xy = EllipseModel().predict_xy(np.linspace(0, 2 * np.pi, num),
                               params=(xc, yc, a, b, angle))
        return xy[:,0],xy[:,1]

class grainApprox():
    
    @classmethod
    def gaussian_fit(cls,y , x,mu=1,sigma=1,amp=1):
        #
        # аппроксимация заданных точек нормальной функцией
        #
        gmodel = Model(grainStats.gaussian)
        res = gmodel.fit(y, x=x, mu=mu,sigma=sigma,amp=amp)
        
        mu=res.params['mu'].value
        sigma=res.params['sigma'].value
        amp=res.params['amp'].value
        
        return mu,sigma,amp

    @classmethod 
    def gaussian_fit_bimodal(cls,y , x, mu1=100,mu2=240,sigma1=30,sigma2=30,amp1=1,amp2=1):
        #
        # аппроксимация заданных точек бимодальной нормальной функцией
        #
        gmodel = Model(grainStats.gaussian_bimodal)
        res = gmodel.fit(y, x=x, mu1=mu1,mu2=mu2,sigma1=sigma1,sigma2=sigma2,amp1=amp1,amp2=amp2)
        
        mus=[res.params['mu1'].value,res.params['mu2'].value]
        sigmas=[res.params['sigma1'].value,res.params['sigma2'].value]
        amps=[res.params['amp1'].value,res.params['amp2'].value]
        
        return mus,sigmas,amps
    
    @classmethod 
    def gaussian_fit_termodal(cls,y , x, mu1=10,mu2=100,mu3=240,sigma1=10,sigma2=30,sigma3=30,amp1=1,amp2=1,amp3=1):
        #
        # аппроксимация заданных точек термодальной нормальной функцией
        #
        gmodel = Model(grainStats.gaussian_termodal)
        res = gmodel.fit(y, x=x, mu1=mu1,mu2=mu2,mu3=mu3,sigma1=sigma1,sigma2=sigma2,sigma3=sigma3,amp1=amp1,amp2=amp2,amp3=amp3)
        
        mus=[res.params['mu1'].value,res.params['mu2'].value,res.params['mu3'].value]
        sigmas=[res.params['sigma1'].value,res.params['sigma2'].value,res.params['sigma3'].value]
        amps=[res.params['amp1'].value,res.params['amp2'].value,res.params['amp3'].value]
        
        return mus,sigmas,amps
