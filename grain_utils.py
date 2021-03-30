import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm


from scipy import ndimage as ndi
from scipy.spatial import distance

from skimage import io
from skimage.color import rgb2gray
from skimage import filters
from skimage.morphology import disk

from PIL import Image, ImageDraw, ImageFilter,ImageOps
import copy
import cv2

from scipy.spatial import ConvexHull

class grainPreprocess(): 

    @classmethod
    def imdivide(cls,image,side):

        width,height = image.size
        sides={'left':0,'right':1}
        shape=[(0,0,width//2,height),(width//2,0,width,height)]
        return image.crop(shape[sides[side]])
    @classmethod
    def combine(cls,image,k=0.5): 
        left_img=cls.imdivide(image,'left')
        right_img=cls.imdivide(image,'right')

        l=k
        r=1-l
        gray=np.array(left_img)*l
        gray+=np.array(right_img)*r
        gray=gray.astype('uint8')
        return gray

    @classmethod
    def do_otsu(cls,img):
        image=rgb2gray(img)

        global_thresh=filters.threshold_otsu(image)
        binary_global = image > global_thresh

        return binary_global
    
    @classmethod
    def image_preprocess(cls,image,combine=0.5):
        combined=cls.combine(image,combine)
        denoised = filters.rank.median(combined, disk(3))
        binary=cls.do_otsu(denoised).astype('uint8')
        grad = filters.rank.gradient(binary, disk(1))
        # у зерен значение пикселя - 0, у тела пустоты - 1, у границы пустоты - 2
        bin_grad=1-binary+grad
        new_image=(bin_grad>0).astype('uint8')*255

        return new_image

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
    def get_line(cls,point1,point2):
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
    def draw_rect(cls,image,point1,point2,r):

        x1,y1=point1[0],point1[1]
        x2,y2=point2[0],point2[1]

        l1,l2=(x2-x1),(y2-y1)

        l_len=(l1**2+l2**2)**0.5
        l_len=int(l_len)

        a=(x1-r*l2/l_len),(y1+r*l1/l_len)
        b=(x1+r*l2/l_len),(y1-r*l1/l_len)

        side=cls.get_line(a,b)

        # a -> c
        lines=np.zeros((side.shape[0],l_len*2,2),dtype='int64')

        for i,left_point in enumerate(side):
            right_point=(left_point[0]+l1),(left_point[1]+l2)
            line_points=cls.get_line(left_point,right_point)
            for j,point in enumerate(line_points):
                lines[i,j]=point

        val=0
        num=0
        for line in lines:
            for point in line:
                if point[0]!=0:
                    x=point[0]
                    y=point[1]

                    val+=image[x,y]
                    num+=1
        return val,num
    
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
            mean=0
            dist=0
        return mean,dist
    
    @classmethod
    def estimate_edges(cls,image,node,corners,position=0,radius=2):
        # вычисление расстояния и среднего значения между 
        # точкой с индексом posion и остальными точками пустоты
        v1=node[position]
        vals=np.zeros((len(node),3))
        if v1!=0:
            for i,v2 in enumerate(node):

                if v2!=0 and v1!=v2  :
                    vals[i,0]=v2
                    point1=corners[v1]
                    point2=corners[v2]
                    mean,dist=cls.mean_pixel(image,point1[0],point2[0],radius)
                    vals[i,1]=mean
                    vals[i,2]=dist

        vals=vals[np.argsort(vals[:,2])]
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

    #                min_val=vals.min(axis=0)[1]
                    for j,val in enumerate(vals):
                        mean=val[1]
                        if abs(mean-0.5)<=eps:
                            v2=val[0]
                            break
                        elif j==vals.shape[0]-1:
                            v2=val[0]
                            break

                  #  print('min_val',min_val)
     #               v2_vals_index=np.where(vals[:,1]==min_val )[0][0]
               #     v2=vals[v2_vals_index][0]
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

class grainShow():
    
    @classmethod
    def img_show(cls,image,N=20,cmap=plt.cm.nipy_spectral):

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
    def draw_edges(image,nodes,corners,color=(51,51,51)):

        im = Image.fromarray(np.uint8(cm.gist_earth(image)*255))
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
        
    