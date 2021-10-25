#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from lmfit.models import Model
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.gridspec as gridspec
from skimage.color import rgb2gray


# In[ ]:


class grainStats():
    
    @classmethod
    def stats_preprocess(cls, array, step):
        # приведенеи углов к кратости, например 0,step,2*step и тд
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
    def ellipse(cls,a,b,angle,xc=0,yc=0,num=50):
        #
        #  возвращает координаты эллипса, построенного по заданным параметрам
        #  по умолчанию центр (0,0)
        #  угол в радианах, уменьшение угла обозначает поворот эллипса по часовой стрелке
        #
        xy = EllipseModel().predict_xy(np.linspace(0, 2 * np.pi, num),
                               params=(xc, yc, a, b, angle))
        return xy[:,0],xy[:,1]


# In[2]:


class grainApprox():
    
    @classmethod
    def gaussian_fit(cls, y , x,mu=1,sigma=1,amp=1):
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
    def gaussian_fit_bimodal(cls, y , x, mu1=100,mu2=240,sigma1=30,sigma2=30,amp1=1,amp2=1):
        #
        # аппроксимация заданных точек бимодальной нормальной функцией
        #
        gmodel = Model(grainStats.gaussian_bimodal)
        res = gmodel.fit(y, x=x, mu1=mu1,mu2=mu2,sigma1=sigma1,sigma2=sigma2,amp1=amp1,amp2=amp2)
        
        mus=[res.params['mu1'].value,res.params['mu2'].value]
        sigmas=[res.params['sigma1'].value,res.params['sigma2'].value]
        amps=[res.params['amp1'].value,res.params['amp2'].value]
        
        return mus,sigmas,amps


# In[3]:


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
    def image_preprocess(cls,image,h,k=0.5):
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


# In[4]:


def from_dist_to_angles(dist):
    a = []
    for i in range(len(dist)):
        for j in range(dist[i]):
            a.append(i)
    return np.array(a)


# In[ ]:


def angles_approx(original_angles_, origs_, images_, names, step, N, font_size=20, save=False, save_name='test.png'):
    #
    # хорошая аппроксимация 
    #
    for i in range(len(images_)):
        
        original_angles = original_angles_[i]
        origs = origs_[i]
        image = images_[i]
        name = names[i]
        
        angles, angles_set, dens_curve = grainStats.stats_preprocess(original_angles, step)
        
        x = np.array(angles_set)
        y = np.array(dens_curve)

        norm=np.sum(y)
        
        mus,sigmas,amps=grainApprox.gaussian_fit_bimodal(y,x)

        x_gauss=range(0,361)
        
        gauss=grainStats.gaussian_bimodal(x_gauss,mus[0],mus[1],sigmas[0],sigmas[1],amps[0],amps[1])
        

     #   plt.legend(text,fontsize=15)
     #   plt.plot(x,y/norm)
     #   plt.hist(angles,bins=100)

        mu1=round(mus[0],2)
        sigma1=round(sigmas[0],2)
        amp1=round(amps[0]/norm,2)
        
        mu2=round(mus[1],2)
        sigma2=round(sigmas[1],2)
        amp2=round(amps[1]/norm,2)

        moda1='\n mu1 = '+str(mu1)+' sigma1 = '+str(sigma1)+' amp1 = '+str(amp1)
        moda2='\n mu2 = '+str(mu2)+' sigma2 = '+str(sigma2)+' amp2 = '+str(amp2)
        val=round(np.log(norm),4)
        total_number='\n количество углов '+ str(int(np.exp(val)))
        text_angle='\n шаг угла '+str(step)+' градусов'
        text=names[i][10:]+moda1+moda2+total_number+text_angle
        
        
        gs = gridspec.GridSpec(2, 2)
        plt.figure(figsize=(N,N))
        
        ax1 = plt.subplot(gs[0, 0]) # row 0, col 0
        ax1.imshow(origs, cmap='gray')
        #ax1.imshow(grainPreprocess.combine(origs,135,1),cmap='gray')
        ax1.set_title('Исходное изображение')

        ax2 = plt.subplot(gs[0, 1]) # row 0, col 1
        ax2.imshow(image, cmap='gray')
        ax2.set_title('Обработанное изображение')

        ax3 = plt.subplot(gs[1, :]) # row 1, span all columns
        ax3.plot(gauss/norm)
        ax3.scatter(x,y/norm)
        ax3.legend([text],fontsize=font_size)
        ax3.set_title('распределение углов связующего вещества', fontsize=font_size)
        ax3.set_ylabel('p(x)', fontsize=font_size)
        ax3.set_xlabel('угол связующего вещества, градусы', fontsize=font_size)

        
       # f,ax=plt.subplots(nrows=2,ncols=2,figsize=(N,N))

       # ax[0,0].imshow(origs[j],cmap='gray')
        
      #  ax[0,1].imshow(image,cmap='gray')
        
      #  ax[1,0].plot(gauss/norm)
      #  ax[1,0].scatter(x,y/norm)
      #  ax[1,0].legend([text],fontsize=font_size)
      #  ax[1,0].set_title('распределение углов связующего вещества', fontsize=font_size)
      #  ax[1,0].set_ylabel('p(x)', fontsize=font_size)
      #  ax[1,0].set_xlabel('угол связующего вещества, градусы', fontsize=font_size)

      #  plt.savefig('крупные_средние_мелкие.png')
        if save:
            plt.savefig(save_name)
        plt.show()

