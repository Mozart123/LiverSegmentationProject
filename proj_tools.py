
# coding: utf-8

# In[ ]:


import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import Augmentor

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


def sitk_show(nda, title=None, margin=0.0, dpi=40):
    #nda = sitk.GetArrayFromImage(img)
    #spacing = img.GetSpacing()
    #figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    figsize = (20,20)
    #extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize) #, dpi=dpi
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()
    
    
def sitk_show_label(nda, title=None, margin=0.0, dpi=40):
    result = nda[:,:,1]
    result[nda[:,:,2] == 1] = 2
    result[nda[:,:,0] == 1] = 3
    
    figsize = (20,20)
    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize) #, dpi=dpi
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(result,extent=extent,interpolation=None, vmin = 0.0, vmax = 3.0)
    
    if title:
        plt.title(title)
    
    plt.show()
    
def random_slice(arrays):
    assert arrays[0].shape == arrays[1].shape
    n_h = np.random.randint(arrays[0].shape[0])
    
    slices = []
    for arr in arrays:
        slices.append(arr[n_h,:,:])
    
    #Augmentation
    p = Augmentor.DataPipeline([slices])
    p.rotate(1, max_left_rotation=5, max_right_rotation=5)
    p.flip_top_bottom(0.5)
    p.zoom_random(0.8, percentage_area=0.8)
    augmented_images = p.sample(1)
    return augmented_images

def get_random_split(folder_addresses, patient_tags):
    num_splits = 10
    patient_id = np.random.randint(len(folder_addresses))
    patient_tag = patient_tags[patient_id]
    data_addr = folder_addresses[patient_id]
    arrays = []
    part_index = np.random.randint(num_splits)
    #print('Selected split: {}/{}'.format(data_addr, patient_tag))
    
    arrays.append(np.load(data_addr + '/' + patient_tag + '_orig_' + str(part_index) + '.npy'))
    arrays.append(np.load(data_addr + '/' + patient_tag + '_lesion_' + str(part_index) + '.npy'))
    arrays.append(np.load(data_addr + '/' + patient_tag + '_liver_' + str(part_index) + '.npy'))
    arrays.append(np.load(data_addr + '/' + patient_tag + '_vessel_' + str(part_index) + '.npy'))
    
    #Rectify 2.0 values
    arrays[1][arrays[1] < 0.0] = 0.0
    arrays[1][arrays[1] > 0.0] = 1.0
    arrays[2][arrays[2] < 0.0] = 0.0
    arrays[2][arrays[2] > 0.0] = 1.0
    arrays[3][arrays[3] < 0.0] = 0.0
    arrays[3][arrays[3] > 0.0] = 1.0
    return arrays
    
#Returns batch_size number of random slices from data, for both input and target
def data_generator(folder_addresses, patient_tags, batch_size):
    n = 0
    arrays = get_random_split(folder_addresses, patient_tags)
    #print(len(arrays))
    while True:
        x_batch = np.zeros((batch_size, 512, 512, 1))
        y_batch = np.zeros((batch_size, 512, 512, 3))
        n+=1
        
        for i_ex in range(batch_size):
            slices = random_slice(arrays)
            #print(slices[0][0].shape)
            x_batch[i_ex, :, :, 0] = slices[0][0]
            y_batch[i_ex, :, :, 0] = slices[0][1]
            y_batch[i_ex, :, :, 1] = slices[0][2]
            y_batch[i_ex, :, :, 2] = slices[0][3] #.reshape((512,512,1)
        
        if n == 25:
            arrays = get_random_split(folder_addresses, patient_tags)
            
        yield x_batch, y_batch

def plot3d(array):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pos = np.where(array==1)
    print(len(pos[0]))
    ax.scatter(pos[0], pos[1], pos[2], cmap='viridis', s = 100) #, c='red'
    plt.show()
