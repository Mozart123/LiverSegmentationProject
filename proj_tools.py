
# coding: utf-8

# In[ ]:

# Functions to split, process and feed the data. Data is too large to fit in memory,
# so it needs to be loaded part by part in training and validation.

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import Augmentor
from tqdm import tqdm

def split_dataset(original_addr_1, lesion_addr_1, liver_addr_1, vessel_addr_1, basename):
    orig_1, origin_orig1, spacing_orig1 = load_itk(original_addr_1)
    lesion_1, origin_lesion1, spacing_lesion1 = load_itk(lesion_addr_1)
    liver_1, origin_lesion1, spacing_lesion1 = load_itk(liver_addr_1)
    vessel_1, origin_lesion1, spacing_lesion1 = load_itk(vessel_addr_1)

    split_data(basename, orig_1, liver_1, lesion_1, vessel_1)

def split_data(base_name, orig, liver, lesion, vessel):
    splits = 10
    stepsize = int(orig.shape[0] / splits)
    indices = np.random.permutation(orig.shape[0])
    orig = orig[indices]
    liver = liver[indices]
    lesion = lesion[indices]
    vessel = vessel[indices]
    
    for i_split in range(splits):
        if i_split < splits - 1:
            np.save(base_name + '_orig_' + str(i_split) + '.npy', orig[i_split * stepsize: (i_split+1) * stepsize,: , :])
            np.save(base_name + '_liver_' + str(i_split) + '.npy', liver[i_split * stepsize: (i_split+1) * stepsize,: , :])
            np.save(base_name + '_vessel_' + str(i_split) + '.npy', vessel[i_split * stepsize: (i_split+1) * stepsize,: , :])
            np.save(base_name + '_lesion_' + str(i_split) + '.npy', lesion[i_split * stepsize: (i_split+1) * stepsize,: , :])
        else:
            np.save(base_name + '_orig_' + str(i_split) + '.npy', orig[i_split * stepsize:,: , :])
            np.save(base_name + '_liver_' + str(i_split) + '.npy', liver[i_split * stepsize:,: , :])
            np.save(base_name + '_vessel_' + str(i_split) + '.npy', vessel[i_split * stepsize:,: , :])
            np.save(base_name + '_lesion_' + str(i_split) + '.npy', lesion[i_split * stepsize:,: , :])

def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    orig_shape = ct_scan.shape
    resized = np.zeros(())
    
    ct_scan = np.flip(ct_scan, axis = 1)
    del(itkimage)
    return ct_scan, origin, spacing#, itkimage


def sitk_show(nda, title=None, margin=0.0, dpi=40, size = (15,15)):
    #nda = sitk.GetArrayFromImage(img)
    #spacing = img.GetSpacing()
    #figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    figsize = size
    #extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize) #, dpi=dpi
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()
    
    
def sitk_show_pred(nda, title=None, margin=0.0, dpi=40):
    #result = nda[:,:,1]
    #result[nda[:,:,2] == 1] = 2
    #result[nda[:,:,0] == 1] = 3
    
    #figsize = (15,15)
    #extent = (0, nda.shape[1], nda.shape[0], 0)
    #fig = plt.figure(figsize=figsize) #, dpi=dpi
    #ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    fig, ax = plt.subplots(1,1, figsize = (15, 15))
    ax.imshow(nda[:,:,0],interpolation=None, vmin = 0.0, vmax = 1.0)
    ax.set_title('prediction')
    #ax[1].imshow(nda[:,:,1],interpolation=None, vmin = 0.0, vmax = 1.0)
    #ax[1].set_title('parankima')
    #ax[2].imshow(nda[:,:,2],interpolation=None, vmin = 0.0, vmax = 1.0) #,extent=extent
    #ax[2].set_title('vessel')
    
    if title:
        plt.title(title)
    
    plt.show()
    
def sitk_show_label(nda, title=None, margin=0.0, dpi=40):
    result = np.zeros((512,512))
    result[nda[0, :, :, 1] == 1] = 1
    result[nda[0, :, :, 0] == 1] = 2
    
    figsize = (15,15)
    fig = plt.figure(figsize=figsize, dpi=dpi) #
    plt.set_cmap("gray")
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    ax.imshow(result, interpolation=None, vmin = 0.0, vmax = 2.0)
    
    if title:
        plt.title(title)
    
    plt.show()
    
def random_slice(arrays):
    assert arrays[0].shape == arrays[1].shape
    n_h = np.random.randint(arrays[0].shape[0])
    
    slices = [arr[n_h] for arr in arrays]
    
    #Augmentation
    p = Augmentor.DataPipeline([slices])
    p.rotate(1, max_left_rotation=5, max_right_rotation=5)
    p.flip_top_bottom(0.5)
    p.zoom_random(0.8, percentage_area=0.8)
    augmented_images = p.sample(1)
    return augmented_images

def get_random_split(folder_addresses, part_counts, layer, keep_all_zeros = False):
    patient_id = np.random.randint(len(folder_addresses))
    data_addr = folder_addresses[patient_id]
    num_splits = part_counts[patient_id]
    
    arrays = []
    part_index = np.random.randint(num_splits-1)
    #print('Selected file: {}, part: {}'.format(data_addr, part_index))
    arrays.append(np.load(data_addr +  '_orig_' + str(part_index) + '.npy'))
    #print('Selected split: {}/{}'.format(data_addr, patient_tag))
    if 'other' in data_addr:
        arr_les_liv = np.load(data_addr  + '_lesion_' + str(part_index) + '.npy')
        arrays.append((arr_les_liv == 2.0).astype('int'))
        arrays.append((arr_les_liv == 1.0).astype('int'))
        arrays.append(((arrays[-2] == 0.0)&(arrays[-1] == 0.0)).astype('int'))
    else:
        arrays.append(np.load(data_addr  + '_lesion_' + str(part_index) + '.npy'))
        arrays.append(np.load(data_addr +  '_liver_' + str(part_index) + '.npy'))
        arrays.append(((arrays[-2] == 0.0)&(arrays[-1] == 0.0)).astype('int'))
            
        #Rectify 2.0 values
        arrays[1][arrays[1] < 0.0] = 0.0
        arrays[1][arrays[1] > 0.1] = 1.0
        arrays[2][arrays[2] < 0.0] = 0.0
        arrays[2][arrays[2] > 0.1] = 1.0
    
    notallzero_idx = None
    if not keep_all_zeros:
        notallzero_idx = np.where(np.sum(np.sum(arrays[1], axis = 1),axis = 1) > 50)[0]
    arrays[3] = arrays[3][notallzero_idx]
    arrays[2] = arrays[2][notallzero_idx]
    arrays[1] = arrays[1][notallzero_idx]
    arrays[0] = arrays[0][notallzero_idx]
    
    if len(notallzero_idx) == 0:
        arrays = get_random_split(folder_addresses, part_counts, layer, keep_all_zeros = False)
    return arrays
    
#Returns batch_size number of random slices from data, for both input and target
def data_generator(folder_addresses, batch_size, filter_img = None, layer = None, isval = False):
    n = 0
    part_counts = []
    for addr in folder_addresses:
        if 'other' in addr:
            part_counts.append(3)
        else:
            part_counts.append(10)
            
    if not isval:
        arrays = get_random_split(folder_addresses, part_counts, layer, keep_all_zeros = False)
        while True:
            x_batch = np.zeros((batch_size, 512, 512, 1))
            y_batch = np.zeros((batch_size, 512, 512, 3))
            n+=1

            for i_ex in range(batch_size):
                allzero = True
                slices = random_slice(arrays)

                if filter_img is not None:
                    slices[0][0] = filter_img.apply(slices[0][0])
                #print(slices[0][0].shape)
                x_batch[i_ex, :, :, 0] = slices[0][0]
                y_batch[i_ex, :, :, 0] = slices[0][1]
                y_batch[i_ex, :, :, 1] = slices[0][2]
                y_batch[i_ex, :, :, 2] = slices[0][3]
                
            if n == 15:
                arrays = get_random_split(folder_addresses, part_counts, layer, keep_all_zeros = False)
                n = 0
            x_batch -= x_batch.mean()
            x_batch /= 2048
            yield x_batch, y_batch
    else:
        while True:
            for i_file, file in enumerate(folder_addresses):
                #print('val file: {}'.format(file))
                for i_part in range(part_counts[i_file]):
                    X_val = np.load(file +  '_orig_' + str(i_part) + '.npy')
                    y_val = None
                    if 'other' in file:
                        y_val_file = np.load(file  + '_lesion_' + str(i_part) + '.npy').reshape((-1,512,512,1))
                        y_val = np.zeros((len(y_val_file), 512, 512, 3))
                        y_val[:, :, :, 0] = (y_val_file == 2.0).astype('int').squeeze() # Lesion
                        y_val[:, :, :, 1] = (y_val_file == 1.0).astype('int').squeeze() # Liver
                        y_val[:, :, :, 2] = ((y_val[:, :, :, 0] == 0.0) & (y_val[:, :, :, 1] == 0.0)).astype('int') # Empty
                    else:
                        y_val_lesion = np.load(file  + '_lesion_' + str(part_index) + '.npy').reshape((-1,512,512,1))
                        y_val_liver = np.load(file +  '_liver_' + str(part_index) + '.npy').reshape((-1,512,512,1))
                        y_val_lesion[y_val_lesion < 0.0] = 0.0
                        y_val_lesion[y_val_lesion > 0.0] = 1.0
                        y_val_liver[y_val_liver < 0.0] = 0.0
                        y_val_liver[y_val_liver > 0.0] = 1.0
                        empties = ((y_val_lesion == 0.0) & (y_val_liver == 0.0)).astype('int')
                        y_val = np.concatenate([y_val_lesion, y_val_liver, empties], axis = -1)
                        
                    i_start = 0
                    i_end = 0
                    for i_ex in range(len(X_val)//batch_size):
                        i_start = batch_size*i_ex
                        i_end = batch_size*i_ex + batch_size
                        X_batch = X_val[i_start:i_end].reshape((-1,512,512,1))
                        
                        # Apply filter
                        if filter_img is not None:
                            for i_inbatch in range(batch_size):
                                    X_batch[i_inbatch, :, :, 0] = filter_img.apply(X_batch[i_inbatch, :, :, 0])
                        X_batch -= X_batch.mean()
                        X_batch /= 2048
                        yield X_batch, y_val[i_start:i_end].reshape((batch_size,512,512,3))
                        
                    if (len(X_val) - i_end) > 0:
                        X_batch = X_val[i_end:].reshape((-1,512,512,1))
                        # Apply filter
                        if filter_img is not None:
                            for i_inbatch in range(len(X_batch)):
                                    X_batch[i_inbatch, :, :, 0] = filter_img.apply(X_batch[i_inbatch, :, :, 0])
                        
                        X_batch -= X_batch.mean()
                        X_batch /= 2048
                        y_batch = y_val[i_end:].reshape((-1,512,512,3))
                        yield X_batch, y_batch
                        
def get_set_len(file_addresses, batch_size):
    part_counts = []
    for addr in file_addresses:
        if 'other' in addr:
            part_counts.append(3)
        else:
            part_counts.append(10)
    set_len = 0
    for i_file, file in tqdm(enumerate(file_addresses)):
        for i_part in range(part_counts[i_file]):
            X = np.load(file +  '_orig_' + str(i_part) + '.npy')
            set_len += len(X)//batch_size
            if len(X) % batch_size != 0:
                set_len += 1
            X = None
            
    return set_len
