 #!/usr/bin/env python
# coding: utf-8

# In[1]:



from __future__ import absolute_import
import io
# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import lmdb
import numpy as np
import caffe
import SimpleITK as sitk
import PIL
import glob
import pydicom
from scipy import ndimage, misc


#############################################################
#  Functions
#############################################################

def _write_to_lmdb(db, key, value):
    """
    Write (key, value) to db
    """
    success = False
    while not success:
        txn = db.begin(write=True)
        try:
            txn.put(key, value)
            txn.commit()
            success = True
        except lmdb.MapFullError:
            txn.abort()
            
            # double the map_size
            curr_limit = db.info()['map_size']
            new_limit = curr_limit*2
            db.set_mapsize(new_limit)   # double it

            
def _save_mean(mean, filename):
    """
    Saves mean to file
    Arguments:
    mean -- the mean as an np.ndarray
    filename -- the location to save the image
    """
    if filename.endswith('.binaryproto'):
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.num = 1
        blob.channels = 1
        blob.height, blob.width = mean.shape
        blob.data.extend(mean.astype(float).flat)
        with open(filename, 'wb') as outfile:
            outfile.write(blob.SerializeToString())
    elif filename.endswith(('.jpg', '.jpeg', '.png')):
        misc.imsave(filename, mean)
    else:
        raise ValueError('unrecognized file extension')



#############################################################
#  Dataset list
#############################################################
"""
Examples of data lists when conversion from B70 to B30 image.
1~9 are train data and 10 is test data.

Converting from 1L1AABHS TO 1S1AABHS
"""

DATA_DIR = "/mnt/DATA/jhy2118/repeatct_1StoL/"
KERNEL_from = "1L1"
KERNEL_to = "1S1"
#File name smooth: E024514S0101I0001
#File name sharp: E024514S0102I0001

train_data_list = ["AABHS", 
"AABSX", 
"AABUY", 
"AABVS", 
"AABWD", 
"AABWV", 
"AACFI", 
"AACFM",
"AACFO"]

test_data_list = ["AACGA", "AACIF", "AACIK"]



#############################################################
#  Train datasets
#############################################################

index = 0

train_image_db = lmdb.open("/mnt/DATA/jhy2118/ct_kernel_conversion/"+KERNEL_from+"to"+KERNEL_to+"/"+"train_image_"+KERNEL_from+"to"+KERNEL_to, map_async=True, max_dbs=0)
train_label_db = lmdb.open("/mnt/DATA/jhy2118/ct_kernel_conversion/"+KERNEL_from+"to"+KERNEL_to+"/"+"train_label_"+KERNEL_from+"to"+KERNEL_to, map_async=True, max_dbs=0)

image_sum = np.zeros((512, 512), 'float64')
image_bias = np.zeros((512, 512), 'uint16')

imageFiles = []
labelFiles = []
    
for data in train_data_list:
    
    imageFiles = sorted(glob.glob(DATA_DIR + KERNEL_from + data +"/E*.dcm"))
    labelFiles = sorted(glob.glob(DATA_DIR + KERNEL_to + data +"/E*.dcm"))
    fileCount = len(imageFiles)
    for c in range(fileCount): #goes thru each dcm file starting with E in the directory
        fileName = imageFiles[c]
        pairName = labelFiles[c]     
        image = sitk.ReadImage(fileName)
        #label = sitk.ReadImage(pairName)
        
        image_in = pydicom.read_file(fileName, force = True) #pydicom method
        label_in = pydicom.read_file(pairName, force = True) #pydicom method
        image_in.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        label_in.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        num_of_slices = image.GetDepth()
        for i in xrange(num_of_slices):
            #nda_img = sitk.GetArrayFromImage(image)[i,:,:]
            #nda_label = sitk.GetArrayFromImage(label)[i,:,:]
            
            nda_img = image_in.pixel_array #pydicom method
            nda_label = label_in.pixel_array #pydicom method

            nda_img = nda_img + image_bias
            nda_label = nda_label + image_bias
            nda_label = nda_label - nda_img

            image_sum += nda_img

            image_datum = caffe.proto.caffe_pb2.Datum()
            image_datum.channels, image_datum.height, image_datum.width = 1, 512, 512
            image_datum.float_data.extend(nda_img.astype(float).flat)
            _write_to_lmdb(train_image_db, str(index+i), image_datum.SerializeToString())

            label_datum = caffe.proto.caffe_pb2.Datum()
            label_datum.channels, label_datum.height, label_datum.width = 1, 512, 512
            label_datum.float_data.extend(nda_label.astype(float).flat)
            _write_to_lmdb(train_label_db, str(index+i), label_datum.SerializeToString())
        index += num_of_slices

image_count = index
print("train image count is " + str(image_count))

# close databases
train_image_db.close()
train_label_db.close()
##
# save mean
mean_image = (image_sum / image_count).astype('uint16')
_save_mean(mean_image, "/mnt/DATA/jhy2118/ct_kernel_conversion/"+KERNEL_from+"to"+KERNEL_to+"/weight/"+"train_image_mean_"+KERNEL_from+"to"+KERNEL_to+".png")
_save_mean(mean_image, "/mnt/DATA/jhy2118/ct_kernel_conversion/"+KERNEL_from+"to"+KERNEL_to+"/weight/"+"train_image_mean_"+KERNEL_from+"to"+KERNEL_to+".binaryproto")



#############################################################
#  Test datasets
#############################################################

testindex = 0

test_image_db = lmdb.open("/mnt/DATA/jhy2118/ct_kernel_conversion/"+KERNEL_from+"to"+KERNEL_to+"/"+"test_image_"+KERNEL_from+"to"+KERNEL_to, map_async=True, max_dbs=0)
test_label_db = lmdb.open("/mnt/DATA/jhy2118/ct_kernel_conversion/"+KERNEL_from+"to"+KERNEL_to+"/"+"test_label_"+KERNEL_from+"to"+KERNEL_to, map_async=True, max_dbs=0)

#image_sum = np.zeros((512, 512), 'float64')
image_bias = np.zeros((512, 512), 'uint16')

imageFiles = []
labelFiles = []

for data in test_data_list:

    imageFiles = sorted(glob.glob(DATA_DIR + KERNEL_from + data +"/E*.dcm"))
    labelFiles = sorted(glob.glob(DATA_DIR + KERNEL_to + data +"/E*.dcm"))
    fileCount = len(imageFiles)
    for c in range(fileCount): #goes thru each dcm file in the directory
        fileName = imageFiles[c]
        pairName = labelFiles[c]
        image = sitk.ReadImage(fileName)
        #label = sitk.ReadImage(pairName)

        image_in = pydicom.read_file(fileName, force = True) #pydicom method
        label_in = pydicom.read_file(pairName, force = True) #pydicom method
        image_in.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        label_in.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        num_of_slices = image.GetDepth()
        for i in xrange(num_of_slices):
            #nda_img = sitk.GetArrayFromImage(image)[i,:,:]
            #nda_label = sitk.GetArrayFromImage(label)[i,:,:]

            nda_img = image_in.pixel_array #pydicom method
            nda_label = label_in.pixel_array #pydicom method

            nda_img = nda_img + image_bias
            nda_label = nda_label + image_bias
            nda_label = nda_label - nda_img

            #image_sum += nda_img

            image_datum = caffe.proto.caffe_pb2.Datum()
            image_datum.channels, image_datum.height, image_datum.width = 1, 512, 512
            image_datum.float_data.extend(nda_img.astype(float).flat)
            _write_to_lmdb(test_image_db, str(testindex+i), image_datum.SerializeToString())

            label_datum = caffe.proto.caffe_pb2.Datum()
            label_datum.channels, label_datum.height, label_datum.width = 1, 512, 512
            label_datum.float_data.extend(nda_label.astype(float).flat)
            _write_to_lmdb(test_label_db, str(testindex+i), label_datum.SerializeToString())

        testindex += num_of_slices

test_image_count = testindex
print("test image count is " + str(testindex))
# close databases
test_image_db.close()
test_label_db.close()


