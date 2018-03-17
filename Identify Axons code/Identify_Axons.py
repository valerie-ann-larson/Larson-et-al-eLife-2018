import re
import glob
import os.path

import numpy as np
import matplotlib.pyplot as mpl
import scipy.misc
from PIL import Image
import skimage
import skimage.segmentation
from skimage import feature

import make_font

def analyze1(binary_file, image_file, min_size=3000, ratio_threshold=2):
    binary = mpl.imread(binary_file)
    if len(binary.shape) == 3:
        binary = binary[:,:,0]
    print 'binary.shape = ', binary.shape
    binary = (binary > 128).astype('uint8')

    font = make_font.load_font(width=20)

    tag = re.sub('_threshold.tif', '', binary_file)
    tag = tag + ('_%d' % min_size)

    data = mpl.imread(image_file)
    if len(data.shape) == 3:
        data = data[:,:,0]
    print 'data.shape = ', data.shape

    print "Labeling"
    labels, num = skimage.measure.label(binary, background=0, connectivity=2, return_num=True)
    labels = labels + 1
    labels2 = skimage.morphology.remove_small_objects(labels, min_size=min_size, connectivity=2)
    (labels2, forward, reverse) = skimage.segmentation.relabel_sequential(labels2)
    nlabels = np.max(labels2) 

    rows = np.zeros(nlabels+1)
    cols = np.zeros(nlabels+1) 
    perims = np.ones(nlabels+1) 
    areas = np.ones(nlabels+1) 

    rows[0] = -100
    cols[0] = -100

    labels = ['%d' % i for i in range(nlabels+1)]
    for i in range(1,nlabels+1):
        g = np.where(labels2 == i)

        r1a = np.min(g[0])
        r1b = np.max(g[0])
        c1a = np.min(g[1])
        c1b = np.max(g[1])
        rows[i] = np.mean(g[0])
        cols[i] = np.mean(g[1])
        print i, rows[i], cols[i]
        thumbnail = labels2[r1a:r1b+1, c1a:c1b+1]
        thumbnail4 = np.zeros((r1b-r1a+5, c1b-c1a+5))
        thumbnail4[2:-2, 2:-2] = thumbnail
        bin_thumb = (thumbnail4 == i)
        thumb_labels, num = skimage.measure.label(1 - bin_thumb, background=0, connectivity=2, return_num=True)
        holes = np.logical_and(thumb_labels != thumb_labels[0,0], thumbnail4 == 0)
        bin_thumb2 = np.logical_or(thumbnail4 == i, holes)
        if np.sum(bin_thumb2) < 500000:
            labels2[r1a:r1b+1, c1a:c1b+1] = labels2[r1a:r1b+1, c1a:c1b+1] + holes[2:-2, 2:-2].astype('int32') * i

        perims[i] = skimage.measure.perimeter(bin_thumb2)
        areas[i] = np.sum(bin_thumb2)

    ratio = (perims / (2 * np.pi)) / np.sqrt(areas / np.pi)
    ok = (ratio < ratio_threshold)
    ok[0] = False

    remap = np.arange(nlabels+1)
    remap[np.logical_not(ok)] = 0;
    labels3 = remap[labels2]

    image = scipy.misc.bytescale(np.dstack((data, data, data)))
    print image.shape
    image[:,:,0] = image[:,:,0] * 0.8 + 0.2 * 255 * (labels2 > 0) 
    image[:,:,1] = image[:,:,1] * 0.8 + 0.2 * 255 * (labels3 > 0) 

    image = label_image(font, image, rows, cols, labels, color=[255,0,0])
    print 'image.shape = ', image.shape
    scipy.misc.imsave(tag+'_labeled_regions_noholes.png', image)

    image = scipy.misc.bytescale(np.dstack((data, data, data)))
    image = label_image(font, image, rows, cols, labels, color=[255,0,0])
    scipy.misc.imsave(tag+'_labeled_regions_simple.png', image)

    with open(tag+'_labeled_regions.txt', 'w') as f:
        f.write('# Column 1: label number\n');
        f.write('# Column 2: area (pixels)\n');
        f.write('# Column 3: row of center of region (pixels)\n');
        f.write('# Column 4: column of center of region (pixels)\n');
        f.write('# Column 5: perimeter (pixel side lengths)\n');
        f.write('# Column 6: (perimeter/2pi) / sqrt(area/pi)  (unitless)\n');
        f.write('# Column 7: OK: (column 6) < %g\n' % ratio_threshold);
        for i in range(1, nlabels+1):
            f.write('%4d %6d %7.2f %7.2f %10.3f  %9.4f  %d\n' % (i, areas[i], rows[i], cols[i], perims[i], \
                ratio[i], ok[i]))
        f.close()

    with open(tag+'_labeled_regions.csv', 'w') as f:
        heading = ('label number', 'area (pixels)', 'row of center of region (pixels)', 
            'column of center of region (pixels)', 'perimeter (pixel side lengths)', 
            '(perimeter/2pi) / sqrt(area/pi) (unitless)', 
            '(column 6) < %g' % ratio_threshold)
        f.write(','.join(['"%s"' % h for h in heading]))
        f.write('\n')
        for i in range(1, nlabels+1):
            t = (i, areas[i], rows[i], cols[i], perims[i], ratio[i],ok[i])
            line = ','.join(['%g' % v for v in t])
            f.write(line)
            f.write('\n')
        f.close()

def label_image(font, image, rows, cols, labels, color=[255,255,255], right_justify=False):
    for r, c, lab in zip(rows, cols, labels):
        r = r.astype('int32')
        c = c.astype('int32')
        label_im = make_font.write_string(font, lab)
        image = make_font.add_to_rgb_image(image, label_im, r, c, color=color, right_justify=right_justify)
    return image
        
def foo1():
    mydir = 'input_files'
    files = glob.glob(os.path.join(mydir, '*threshold.tif'))
    print files

    for f in files:
        binary_file = f
        image_file = re.sub('_threshold', '', f)
        assert(image_file != f)
        analyze1(binary_file, image_file, min_size=500, ratio_threshold=2)


if __name__ == "__main__":
    foo1()


