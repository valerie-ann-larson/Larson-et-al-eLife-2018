import re
import glob

import matplotlib.pyplot as mpl
import matplotlib
import numpy as np
import stfio
import scipy.optimize


def foo():
    abf_file = 'data4/16n15003 0196 N1 ACSF baseline.abf'
    rec = stfio.read(abf_file)

    data = np.array(rec[0])
    print data.shape

    for i in range(data.shape[0]):
        mpl.plot(data[i,:])
    mpl.show()


def load_data(abf_file='data2/16o27060.abf',offset=False,norm=True):
    rec = stfio.read(abf_file)
    data = np.array(rec[0])
    t = rec.dt * np.arange(data.shape[1]) 
    if offset:
        t0 = 327.88
        if re.search('stim.abf', abf_file):
            t0 = 0.215
        if re.search('CAP.abf', abf_file):
            t0 = 0.215 + 0.31
        t = t - t0
    if norm:
        g = np.where((t >= -0.20) * (t <= -0.05))[0]
        for i in range(data.shape[0]):
            data[i,:] = data[i,:] - np.mean(data[i,g])
        
    return t, data


def write_vector(filename, vec):
    with open(filename, 'w') as f:
        for v in vec:
            f.write('%g\n' % v)
        f.close()


def write_data(filename, data, form=None):
    if form == None:
        form = ' '.join(['%g' for i in range(data.shape[1])])
    with open(filename, 'w') as f:
        for i in range(data.shape[0]):
            f.write((form+'\n') % tuple(data[i,:]))
        f.close()

def draft1():
    def get_means(t, data, time_range=(-100, 100)):
        indices = np.where((t >= time_range[0]) * (t <= time_range[1]))[0]
        t = t[indices]
        data = data[:,indices]
        print 'time range = ', np.amin(t), np.amax(t)

        means = np.zeros(data.shape[0])
        medians = np.zeros(data.shape[0])
        areas = np.zeros(data.shape[0])
        peaks = np.amax(data, axis=1)

        for i in np.arange(data.shape[0]):
            norm = np.sum(data[i,:])
            means[i] = np.sum(data[i,:] * t) / norm

            cumulative = np.cumsum(data[i,:])
            g = np.where(cumulative < 0.5 * cumulative[-1])[0]
            medians[i] = t[g[-1]]

            areas[i] = np.mean(data[i,:]) * (t[-1] - t[0])

        return means, medians, areas, peaks, (np.amin(t), np.amax(t))


    def get_fwhm(t, data, time_range=(-100, 100)):
        indices = np.where((t >= time_range[0]) * (t <= time_range[1]))[0]
        t = t[indices]
        data = data[:,indices]
        print 'time range = ', np.amin(t), np.amax(t)

        fwhm = np.zeros(data.shape[0])
        heights = np.zeros(data.shape[0])
        starts = np.zeros(data.shape[0])
        ends = np.zeros(data.shape[0])

        for i in np.arange(data.shape[0]):
            peak_index = np.argmax(data[i,:])
            peak = np.mean(data[i,peak_index-2:peak_index+3])
            cut_height = 0.5 * peak
            g = np.where(data[i,:] >= cut_height)[0]
            start_time = np.amin(t[g])
            end_time = np.amax(t[g])
            fwhm[i] = end_time - start_time
            heights[i] = cut_height
            starts[i] = start_time
            ends[i] = end_time

        assert(np.amin(starts) >= t[0])
        assert(np.amax(ends) <= t[-1])

        return fwhm, heights, starts, ends, (np.amin(t), np.amax(t))

    labels = ['CAP', 'baseline', 'stim', 'recovery']
    time_range = np.array((0.41, 6)) - 0.215
    fwhm_time_range = (time_range[0], 10)

    for label in labels:
        files = glob.glob('*/*%s.abf' % label)
        files.sort()
        for filename in files:
            tag = re.sub('.abf', '', filename)
            print tag
            print label, filename
            t, data = load_data(filename,offset=True,norm=True)
            means, medians, areas, peaks, actual_time_range = get_means(t, data, time_range=time_range)

            filename = tag + '_%5.3f_%5.3f_mean_ms.dat' % actual_time_range
            write_vector(filename, means)
            filename = tag + '_%5.3f_%5.3f_median_ms.dat' % actual_time_range
            write_vector(filename, medians)
            filename = tag + '_%5.3f_%5.3f_area_mVms.dat' % actual_time_range
            write_vector(filename, areas)
            filename = tag + '_%5.3f_%5.3f_peak_mV.dat' % actual_time_range
            write_vector(filename, peaks)

        """
            if False:
                mpl.plot(means, color)
                mpl.plot(medians, color, ls='--', lw=2)
        if False:
            mpl.title(my_dir)
            mpl.show()
        """

if __name__ == "__main__":
    draft1()
  




