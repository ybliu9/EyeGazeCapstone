import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


def remove_background(img=None, method='binary',thresh=150, sharpen=True, tozero_thresh=None):
    """
    processes the input image by thresholding, sharpening, or recoloring, 
    and returns a binary/grayscale image
    
    arguments
    -----------
    img             array, the input RGB image as a numpy array with shape (height, width, channel)
    method          str, method of image processing, e.g. 'binary', 'tozero', 'OTSU', 'adaptive'
                    -- 'binary' for binary thresholding --> generates binary image
                    -- 'tozero' for tozero thresholding --> generates grayscale image
                    -- 'OTSU' for OTSU thresholding --> generates binary image
                    -- 'adaptive' for adaptive thresholding --> generates binary image
    thresh          int, range [0,255], the threshold value for binary image, default = 150
    sharpen         bool, whether to sharpen the input image, default = True
    tozero_thresh   tuple/list, the low and high thresholds applied ONLY when method='tozero', 
                    e.g. (100, 200) would change values under 100 to 0 (black) and change values 
                    over 200 to 255 (white), default = None
    
    
    returns
    -----------
    _params         str, a string containing input parameter values
    thresh_values   array, the processed grayscale image as a numpy array with shape (height, width)
    
    """
    # make sure the input image is valid
    assert img is not None and method in ['binary', 'tozero', 'OTSU', 'adaptive']
    
    #sharpen (not applied to adaptive shresholding)
    if sharpen and method != 'adaptive':
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    
    #grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[-1] == 3 else img
    
    func = cv2.THRESH_BINARY_INV
    
    if method == 'binary':
        ret, thresh_values = cv2.threshold(gray,thresh, 255, func)
    elif method == 'tozero':
        ret, thresh_values = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY+cv2.THRESH_TOZERO)
    elif method == 'OTSU':
        ret, thresh_values = cv2.threshold(gray, thresh, 255, cv2.THRESH_OTSU)
    elif method == 'adaptive':
        thresh_values = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, func, 5, 10)
    
    # Further process each row for images with multiple background colors
    for row in range(thresh_values.shape[0]):
        if method == 'tozero':
            # tozero shresholding generates grayscale image instead of binary image
            ret, r = cv2.threshold(thresh_values[row,:].reshape(-1,1),thresh,255,cv2.THRESH_BINARY)
            # count occurrences of color values
            unique, counts = np.unique(r.reshape(-1,), return_counts=True)
            #thresh_values[row,:] = 0.7*r.reshape(-1,)+0.3*thresh_values[row,:]
            if tozero_thresh is not None and len(tozero_thresh) == 2 :
                l,h = tozero_thresh
                thresh_values[row,:] = np.where(thresh_values[row,:] >= h, 255, thresh_values[row,:])
                thresh_values[row,:] = np.where(thresh_values[row,:] <= l, 0, thresh_values[row,:])
        else: 
            #count occurrences of color values
            unique, counts = np.unique(thresh_values[row,:], return_counts=True)
            
        dic = dict(zip(unique, counts))
        
        # if this row has no characters, skip
        if 0 not in dic.keys(): continue    #thresh_values[row,:] = 0*thresh_values[row,:]
        # if this row has white characters in a dark background, flip the colors
        elif 255 not in dic.keys() or dic[0] > dic[255]:
            # inverse colors
            thresh_values[row,:] = 255-thresh_values[row,:]
            
    
    _params = method+', shreshold='+ str(thresh) + int(sharpen)*', sharpened'
    #return a parameter string and a binary/grayscale image with white background
    return _params, thresh_values

def get_structure(thresh_values=None, binary=True):
    """
    gets structures of the binary/grayscale input image 
    and returns a binary/grayscale image
    
    arguments
    -----------
    thresh_values   array, the input binary/grayscale image as a numpy array with shape (height, width)   
    binary          bool, force to return a binary image if binary = True, default = True
    
    returns
    -----------
    thresh_values   array, the processed image with structural lines, as a numpy array with 
                    shape (height, width)
    
    """
    # make sure the input image is valid
    assert thresh_values is not None
    
    row, col = thresh_values.shape
    thresh_values = np.where(thresh_values <= 150, 0, thresh_values)
    thresh_values = np.where(thresh_values >= 200, 255, thresh_values)
    
    v_lines = []
    # GET VERTICAL BORDER LINES
    for c in range(col):
        unique, counts = np.unique(thresh_values[:,c], return_counts=True)
        dic = dict(zip(unique, counts))
        #print(dic)
        if 0 not in dic.keys() or dic[0] <= 1: #thresh_values[:,c] = 0
            v_lines.append(c)

    # GET HORIZONTAL BORDER LINES
    for r in range(row):
        for c in range(col//10, col, 5):
            # if the row has a semi-structured/incomplete border line with length of at
            # least 1/10 of the width, complete the line 
            if all(0 == thresh_values[r,(c-col//10):c]):
                thresh_values[r-2:r,:] = 0
                break
        unique, counts = np.unique(thresh_values[r,:], return_counts=True)
        dic = dict(zip(unique, counts))
        # if the row has no characters, draw a black line 
        if 0 not in dic.keys(): 
            thresh_values[r,:] = 0
        # if the entire row is black, skip
        elif dic[0] == col: continue
        # if the row has characters, hide the characters with light colors
        else:
            thresh_values[r,:] = np.where(thresh_values[r,:] == 0, 220, thresh_values[r,:])
    
    #print(np.unique(thresh_values, return_counts=True))
            
    for c in v_lines:
        thresh_values[:,c-1:c+1] = 0
        
    if binary:
        ret, thresh_values = cv2.threshold(thresh_values, 160, 255, cv2.THRESH_BINARY)
        
    return thresh_values



def get_borders(structured=None, kernel0=2, kernel1=7, 
                erode0_iter=1, erode1_iter=1, dilate0_iter=2, dilate1_iter=3,
                stripe0=4, stripe1=6, plot=True, img=None, img_name=None):
    """
    gets structures of the binary/grayscale input image 
    and returns a binary/grayscale image
    
    arguments
    -----------
    structured      array, the input binary/grayscale image as a numpy array with shape (height, width) 
    kernel0         int, range [1,height), the kernel size for cv2.MORPH_RECT. Used for horizontal border 
                    lines, default = 2
                    -- Tip: for short tables or tables with many lines, use a smaller kernel0
    kernel1         int, range [1,width), the kernel size for cv2.MORPH_RECT. Used for vertical border lines, 
                    default = 7
                    -- Tip: for wide tables or tables with a few columns, use a larger kernel1
    erode0_iter     int, range [1,inf), the number of iterations for horizontal erosion, default = 1
    erode1_iter     int, range [1,inf), the number of iterations for vertical erosion, default = 1
    dilate0_iter    int, range [1,inf), the number of iterations for horizontal dilation, default = 2
    dilate1_iter    int, range [1,inf), the number of iterations for vertical dilation, default = 3
    stripe0         int, range [1,inf), the minimum width for a horizontal line to be considered as a 
                    border, default=4
    stripe1         int, range [1,inf), the minimum width for a vertical line to be considered as a 
                    border, default=6
    plot            bool, display results if plot = True, default = True
    
    returns
    -----------
    coords          array, the processed image with structural lines, as a numpy array with 
                    shape (height, width)
    
    """
    
    # make sure the input image is valid
    assert structured is not None
    
    r,c = structured.shape
    
    #recognize horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,kernel0))
    eroded = cv2.erode(structured, kernel, iterations = erode0_iter)
    dilatedrow = cv2.dilate(eroded, kernel, iterations = dilate0_iter)
    
    #recognize vertical lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel1,1))
    eroded = cv2.erode(structured, kernel, iterations = erode1_iter)
    #change iterations based on image quality
    dilatedcol = cv2.dilate(eroded, kernel, iterations = dilate1_iter)
    
    #recognize intersections
    bitwiseAnd = cv2.bitwise_and(dilatedcol,dilatedrow)

    #merge
    #merge = cv2.add(dilatedcol, dilatedrow)
    
    #get intersection coordinates
    xs, ys = np.where(bitwiseAnd>1)
    
    xs.sort()
    ys.sort()
    #print(np.unique(xs))
    #print(np.unique(ys))


    
    listx=[0] 
    for i in range(1,len(xs)):
        if(xs[i]-xs[i-1] >= stripe0):
            listx.append((xs[i]+xs[i-1])//2)
    listx.append(r-1) #add the right edge
    
    listy=[0] 
    for i in range(1,len(ys)):
        if(ys[i]-ys[i-1] >= stripe1):
            listy.append(ys[i]-stripe1)
    listy.append(c-1) #add the bottom edge


    # coords = np.array([[x,y] for x in listx for y in listy])
    coords = list()
        
    print('Row coordinates: ', listx)
    print('Column coordinates: ', listy)
    
    
    if plot: 
        assert img is not None and img_name is not None
        plots_ = dict()
        if img is not None:
            plots_['original image with border coordinates'] = img
        subtitle = img_name + ': ' if img_name is not None else ""
        plots_['intersections'] = bitwiseAnd
        plots_['dilated rows'] = dilatedrow
        plots_['dilated columns'] = dilatedcol
        #plots_['merged'] = merge
        
        fig, axes = plt.subplots(len(plots_), 1, sharex=False, #True,
                         gridspec_kw=(dict(hspace=0.3,wspace=0.1)), figsize=(12,r//35*len(plots_))
                         )
        d_ = list(plots_.items())
        for i in range(len(plots_)):
            axes[i].imshow(cv2.cvtColor(d_[i][1], cv2.COLOR_BGR2RGB))
            axes[i].set_title(subtitle + d_[i][0])

        axes[0].scatter(coords.T[1], coords.T[0], s=15, c='r')
        plt.show()

    
    return listx, listy

def main():
    # EXAMPLE: TABLE CELL
    ############################################################################################################################
    # Sample plots with two example cells

    #example header / content cell images as input
    header_cell1 = img[10:20, 400:480]
    plots1 = dict()
    plots1['original'] = header_cell1

    content_cell1 = img[30:45, 0:130]            
    plots2 = dict()
    plots2['original'] = content_cell1




    # try different methods
    for s in [True]: #[False, True]
        for m in ['binary','tozero','OTSU','adaptive']:
            for th in [150]: #[120, 125, 130]
                title, im = remove_background(header_cell1, method=m, thresh=th, sharpen=s)
                plots1.update({title:im})
                title, im = remove_background(content_cell1, method=m, thresh=th, sharpen=s)
                plots2.update({title:im})

    #plotting
    fig, axes = plt.subplots(len(plots1), 2, sharex=False, #True,
                            gridspec_kw=(dict(hspace=1,wspace=0.1)), figsize=(12,1.5*len(plots1))
                            )
    d = list(plots1.items())
    d2 = list(plots2.items())
    for i in range(len(plots1)):
        axes[i][0].imshow(cv2.cvtColor(d[i][1], cv2.COLOR_BGR2RGB))
        axes[i][0].set_title('Header cell: '+d[i][0])
        axes[i][1].imshow(cv2.cvtColor(d2[i][1], cv2.COLOR_BGR2RGB))
        axes[i][1].set_title('Content cell: '+d2[i][0])

    plt.show()



    # EXAMPLE: FULL TABLE
    #############################################################################################################################
    # read image, modify code as needed
    i = 2

    img_name = images[i].split('.')[0]
    img=cv2.imread(cropped_tables[i])
    print("image: %s"%img_name)
    row,col,channel=img.shape   
    print("height: %d, width: %d, channels: %d"%(row, col, channel))
    print("pixels: %d" % (img.size)) #number of pixels
    print("dtype: %s" % (img.dtype))   #dtype of image, normally uint8
    #############################################################################################################################
    plots_=dict()
    plots_['original'] = img

    # try different methods
    for s in [True]: #[False, True]
        for m in ['binary','tozero','OTSU']: #'adaptive'
            for th in [150]: #[120, 125, 130]
                tozero_thresh = [100,200] if m == 'tozero' else None
                title, im = remove_background(img, method=m, thresh=th, sharpen=s,tozero_thresh = tozero_thresh)
                plots_.update({title:im})
    # plot
    fig, axes = plt.subplots(len(plots_), 1, sharex=False, #True,
                            gridspec_kw=(dict(hspace=0.3,wspace=0.1)), figsize=(12,3*len(plots_))
                            )
    d_ = list(plots_.items())
    for i in range(len(plots_)):
        axes[i].imshow(cv2.cvtColor(d_[i][1], cv2.COLOR_BGR2RGB))
        axes[i].set_title(img_name+': '+d_[i][0])

    plt.show()


    #EXAMPLE: get structure and border lines
    ################################################################################################################
    # read image, modify code as needed
    i = 4

    img_name = images[i].split('.')[0]
    img=cv2.imread(cropped_tables[i])
    print("image: %s"%img_name)
    row,col,channel=img.shape   
    print("height: %d, width: %d, channels: %d"%(row, col, channel))
    print("pixels: %d" % (img.size)) #number of pixels
    print("dtype: %s" % (img.dtype))   #dtype of image, normally uint8
    #################################################################################################################

    # Recommend:
    # Use remove_background(img, method='tozero',tozero_thresh =[100,200]) for higher quality

    #step1: remove background
    title, im = remove_background(img, method='tozero',tozero_thresh =[100,200])
    #step2: get the table structure
    thresh_values = get_structure(im, binary=True)
                
    plt.imshow(cv2.cvtColor(thresh_values, cv2.COLOR_BGR2RGB))
    plt.title('Table structure')
    plt.show()



    #step3: get coordinates for border lines
    coords = get_borders(thresh_values, plot=True, kernel1 = 7, erode0_iter=1, erode1_iter=1, 
                        dilate0_iter=2, dilate1_iter=3,
                        stripe1=6,
                        img=img, img_name=img_name)


if __name__ == "__main__":
    main()

