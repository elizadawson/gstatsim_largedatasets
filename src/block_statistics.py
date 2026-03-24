# Author: Michael Field
# Updated: Feb 12 2026

import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar

def block_reduce_par(xbin_edges, ybin_edges, coords, value, metrics=['mean'], quantiles=None, progress=True):
    dx = np.abs(xbin_edges[1]-xbin_edges[0])
    dy = np.abs(ybin_edges[1]-ybin_edges[0])
    x_center = xbin_edges[:-1]+dx/2
    y_center = ybin_edges[:-1]+dy/2

    shp = (y_center.size, x_center.size)

    if quantiles is None:
        result = np.full((len(metrics), *shp), np.nan)
    else:
        result = np.full((len(metrics)+len(quantiles), *shp), np.nan)

    xbin_inds = np.digitize(coords[0], xbin_edges)-1
    ybin_inds = np.digitize(coords[1], ybin_edges)-1

    bins = np.stack((xbin_inds.flatten(), ybin_inds.flatten())).T

    if quantiles is None:
        result = np.full((len(metrics), *shp), np.nan)
    else:
        result = np.full((len(metrics)+len(quantiles), *shp), np.nan)

    if progress==True:
        with ProgressBar(total=np.unique(bins[:,0]).size) as progress:
            result = block_reduce_jit(bins, shp, result, coords, value, metrics, quantiles, progress)
    else:
        result = block_reduce_jit(bins, shp, result, coords, value, metrics, quantiles, None)

    return result, (x_center, y_center)

@njit(parallel=True)
def block_reduce_jit(bins, shp, result, coords, value, metrics=['mean'], quantiles=None, progress_proxy=None):

    xb_uniq = np.unique(bins[:,0])
    yb_uniq = np.unique(bins[:,1])

    update_progressbar = progress_proxy is not None
    
    for i in prange(xb_uniq.size):
        xbin = xb_uniq[i]
        
        # check for out of bin bounds
        if (xbin==shp[1]) | (xbin==-1):
            if update_progressbar:
                progress_proxy.update(1)
            continue
        inds1 = bins[:,0]==xbin
        bins1 = bins[inds1,:]
        value1 = value[inds1]
        
        for j in range(yb_uniq.size):
            ybin = yb_uniq[j]
            
            # check for out of bin bounds
            if (ybin==shp[0]) | (ybin==-1):
                continue
            inds2 = bins1[:,1]==ybin
            value2 = value1[inds2]
            
            count = value2.size
            if count > 0:
                for k, metric in enumerate(metrics):
                    if metric == 'count':
                        result[k,ybin,xbin] = count
                    elif metric == 'mean':
                        result[k,ybin,xbin] = np.nanmean(value2)
                    elif metric == 'median':
                        result[k,ybin,xbin] = np.nanmedian(value2)
                    elif metric == 'std':
                        result[k,ybin,xbin] = np.nanstd(value2)
                    elif metric == 'var':
                        result[k,ybin,xbin] = np.nanvar(value2)
                    elif metric == 'skew':
                        result[k,ybin,xbin] = np.nanmean((value2-np.nanmean(value2))**3) / np.nanmean((value2-np.nanmean(value2))**2)**1.5
                    elif metric == 'kurtosis':
                        result[k,ybin,xbin] = np.nanmean(((value2 - np.nanmean(value2)) / np.nanstd(value2))**4) - 3
                if quantiles is not None:
                    for k, q in enumerate(quantiles):
                        result[k+len(quantiles),ybin,xbin] = np.nanquantile(value2, q)

        if update_progressbar:
            progress_proxy.update(1)
        
    return result

def block_reduce(xbin_edges, ybin_edges, coords, value, metrics=[np.mean], quantiles=None):
    dx = np.abs(xbin_edges[1]-xbin_edges[0])
    dy = np.abs(ybin_edges[1]-ybin_edges[0])
    x_center = xbin_edges[:-1]+dx/2
    y_center = ybin_edges[:-1]+dy/2

    shp = (y_center.size, x_center.size)

    if quantiles is None:
        result = np.full((len(metrics), *shp), np.nan)
    else:
        result = np.full((len(metrics)+len(quantiles), *shp), np.nan)

    xbin_inds = np.digitize(coords[0], xbin_edges)-1
    ybin_inds = np.digitize(coords[1], ybin_edges)-1

    bins = np.stack((xbin_inds.flatten(), ybin_inds.flatten())).T
    
    xb_uniq = np.unique(bins[:,0])
    yb_uniq = np.unique(bins[:,1])
    
    for i in tqdm(range(xb_uniq.size)):
        xbin = xb_uniq[i]
        
        # check for out of bin bounds
        if (xbin==shp[1]) | (xbin==-1):
            continue
        inds1 = bins[:,0]==xbin
        bins1 = bins[inds1,:]
        value1 = value[inds1]
        
        for j in range(yb_uniq.size):
            ybin = yb_uniq[j]
            
            # check for out of bin bounds
            if (ybin==shp[0]) | (ybin==-1):
                continue
            inds2 = bins1[:,1]==ybin
            value2 = value1[inds2]
            
            count = value2.size
            if count > 0:
                for k, metric in enumerate(metrics):
                    result[k,ybin,xbin] = metric(value2)
                if quantiles is not None:
                    for k, q in enumerate(quantiles):
                        result[k+len(quantiles),ybin,xbin] = np.nanquantile(value2, q)
        
    return result, (x_center, y_center)