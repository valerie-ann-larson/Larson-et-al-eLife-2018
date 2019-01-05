import numpy as np
import matplotlib.pyplot as mpl
import scipy.misc
import os

def foo2():
    fig = mpl.figure(figsize=(1,2))
    for i in range(128):
        ax = mpl.axes([0, 0, 1, 1])
        #mpl.text(0, 0.20, chr(i), size=120, family='monospace') 
        mpl.text(0, 0.20, chr(i), size=120, family='sans-serif') 
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        mpl.savefig('ascii/test%3.3d.png' % i, dpi=120)
        mpl.clf()
        print i


def foo3():
    font = np.zeros((240, 120, 128), dtype='uint8')
    for i in range(128):
        file = 'ascii/test%3.3d.png' % i
        image = scipy.misc.imread(file)
        #print image.shape
        #font[:,:,i] = 255 - image[:,:,0]
        font[:,:,i] = 255 - image[:,0:120,0]
        #font[:,:,i] = 255 - image[:,0:150,0]

    print font.shape
    np.save('font_240_120_128.npy', font, allow_pickle=False)


def load_font(width=None):
    font = np.load('font_240_120_128.npy', allow_pickle=False)
    #print font.shape

    if width != None:
        font = resize_font(font, width)

    return font


def write_string(font, string):
    #print len(string)

    height = font.shape[0]
    width = font.shape[1] 
    #print height, width

    image = np.zeros((height, width * len(string)), dtype='uint8')
    for i, c in enumerate(string):
        image[:,i*width:(i+1)*width] = font[:,:,ord(c)]
        
    return image


def write_array(font, array):
    height = font.shape[0]
    width = font.shape[1] 
    #print height, width

    rows = len(array)
    nchar = max(map(len, array))
    
    image = np.zeros((height * rows, width * nchar), dtype='uint8')
    for r in range(len(array)):
        for i, c in enumerate(array[r]):
            image[r*height:(r+1)*height,i*width:(i+1)*width] = font[:,:,ord(c)]
        
    return image

def rebin_image(a, shape):
    """ 
    http://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    """
    if len(shape) == 2:
        if a.shape[0] > shape[0]:
            assert(a.shape[1] >= shape[1])
            sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
            return a.reshape(sh).mean(3).mean(1)
        elif a.shape[0] < shape[0]:
            assert(a.shape[1] <= shape[1])
            return np.repeat(np.tile(a, shape[0]//a.shape[0]), shape[1]//a.shape[1]).reshape(shape)
        else:
            assert(a.shape[1] == shape[1])
            return a
    elif len(shape) == 1:
        if a.shape[0] >= shape[0]:
            sh = shape[0], a.shape[0]//shape[0]
            return a.reshape(sh).mean(1)
        else:
            raise NotImplementedError        
    else:
        assert(len(shape) == 3)
        sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1], a.shape[2]
        return a.reshape(sh).mean(3).mean(1)


def resize_font(font, width):
    h = font.shape[0]
    w = font.shape[1] 

    height = (h * width) / w
    print h, w
    print height, width
    
    font2 = rebin_image(font, (height, width, 128)).astype('uint8')
    return font2


def overlap1d(len1, len2, row):
    r1range = np.arange(len1)
    r2range = np.arange(len2)
    g1 = (r1range >= row) * (r1range < row + len2)
    g2 = (r2range >= -row) * (r2range < -row + len1)

    gg1 = np.where(g1)
    gg2 = np.where(g2)
    #print gg1
    #print gg2

    if len(gg1[0]) > 0:
        assert(len(gg1[0]) == len(gg2[0]))
        r1a = np.amin(gg1[0])
        r1b = np.amax(gg1[0]) + 1
        r2a = np.amin(gg2[0])
        r2b = np.amax(gg2[0]) + 1
    else:
        assert(len(gg2[0]) == 0)
        r1a = 0
        r1b = 0
        r2a = 0
        r2b = 0

    return r1a, r1b, r2a, r2b


def overlap(im1, im2, row, col):
    """
    place im2 in im1 at row,col, so that
    im1[row,col] = im2[0,0]
    Calculate overlap, so that 
    im1[r1a:r1b,c1a:c1b] = im2[r2a:r2b, c2a:c2b]
    """
    h1, w1 = im1.shape[0:2]
    height, width = im2.shape[0:2]

    r1a, r1b, r2a, r2b = overlap1d(h1, height, row)
    c1a, c1b, c2a, c2b = overlap1d(w1, width, col)

    return r1a, r1b, r2a, r2b, c1a, c1b, c2a, c2b


def add_to_rgb_image(image, text, row, col, color=[255, 255, 255], right_justify=False):
    height = text.shape[0] 
    width = text.shape[1] 

    if right_justify:
        col = col - width

    if len(image.shape) == 2:
        r1a, r1b, r2a, r2b, c1a, c1b, c2a, c2b = overlap(image, text, row, col) 
        image[r1a:r1b, c1a:c1b] = \
            (text[r2a:r2b, c2a:c2b] / 255.0) * color[0] + \
            (1 - text[r2a:r2b, c2a:c2b] / 255.0) * image[r1a:r1b, c1a:c1b]
    else:
        assert(len(image.shape) == 3)
        r1a, r1b, r2a, r2b, c1a, c1b, c2a, c2b = overlap(image, text, row, col) 
        for i in range(3):
            image[r1a:r1b, c1a:c1b, i] = \
                (text[r2a:r2b, c2a:c2b] / 255.0) * color[i] + \
                (1 - text[r2a:r2b, c2a:c2b] / 255.0) * image[r1a:r1b, c1a:c1b, i]

    return image

if __name__ == "__main__":
    foo2()
    foo3()

