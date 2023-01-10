import numpy as np
import os
import glob
import rasterio as rio
import sys
DEF_WEIGHTS = np.array([0.4268,2.8883,2.1803,1.0000,0.4450,2.5977,0.2878])
DEF_LIST_LABELS = [0,1,2,3,4,5,6,255]
#correspondances={0:0,1:1,2:2,3:3,4:4,5:5,6:4,255:6}
correspondances={0:0,1:1,2:2,3:3,4:4,5:5,6:6,255:7}
DEF_DATASET='/home/tcorpetti/SENTINEL-2/TMP4TCo/umask_training_samples_v1/Validation/masks/'
DEF_NODATA=9999999
def compute_weights_from_dict(dict):
    """
    Example of dict : dict={0 : 29548,1 : 8900,2:234}
    :param dict: dictionnary of number of data in classes
    :return: weight to put in categorical cross entropy
    """
    labels=[*dict.keys()]
    values=[*dict.values()]
    return np.sum(values) / np.multiply(len(labels) ,values)

def create_dict(list_labels, dataset_mask,correspondances,nodata=DEF_NODATA):
    dict = {}
    for i in range(len(list_labels)):
        dict[i] = 0
    list_masks=glob.glob('%s/*tif'%dataset_mask)
    for i in range(len(list_masks)):
        tmp=rio.open(list_masks[i]).read()
        tmp=np.where(tmp==nodata,np.nan,tmp)
        #val_label,nb_label=np.unique(tmp,return_counts=True)
        val_label,nb_label=np.unique(tmp[~np.isnan(tmp)],return_counts=True)
        print('image ',list_masks[i])
        print('val_label',val_label)
        print('nb_label',nb_label)
        for j in range(len(val_label)):
            dict[correspondances[val_label[j]]]+=nb_label[j]
        toprint = "[%.5d]"% i
        sys.stdout.write(toprint + chr(13))

    return dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute weights for NN classification based on weighted categorical cross entropy")
    parser.add_argument("--list_labels", nargs="+", type=int, default=DEF_LIST_LABELS,
                        help="list of (integers) labels (default : %s)"%DEF_LIST_LABELS)
    #parser.add_argument("--weights", nargs="+", default=DEF_WEIGHTS,type = float,
    #                    help="associated weights (default : %s)"%DEF_WEIGHTS)
    parser.add_argument("--dataset", type=str, default=DEF_DATASET,
                        help="dataset with masks tif files (def : %s)"%DEF_DATASET)
    parser.add_argument("--nodata", type=int, default=DEF_NODATA,
                        help="value for nodata (def : %d)"%DEF_NODATA)
    args = parser.parse_args()
    #weights=args.weights
    dataset=args.dataset
    nodata=args.nodata
    weights = 0
    list_labels=args.list_labels
    print('list labels')
    print(list_labels)
    print('compute weights associated with labels')
    correspondances={0: 0,1: 1, 2: 2, 3: 3, 4: 4}
    dictio = create_dict(list_labels, dataset, correspondances,nodata)
    weights = compute_weights_from_dict(dictio)
    print('Final weights : ', weights)
    for i in range(len(weights)):
        print('weight for label %d %.13f'%(list_labels[i],weights[i]))




# Combine generators
# https://stackoverflow.com/questions/46313525/how-do-i-combine-two-keras-generator-functions
