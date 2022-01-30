import time
import numpy as np
from PIL import Image
import os,sys,platform
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def image_to_numpy(image_path,show=True):
    image = Image.open(image_path)
    image.load()
    if show:
        print('------ Initial Image -------')
        plt.imshow(image)
        plt.show()
    else:
        pass
    image_array = np.array(image)
    initial_image_copy = np.copy(image_array)
    height, width, pixel_vals = image_array.shape
    image_array = np.reshape(image_array,(height*width,pixel_vals)).astype(np.int16)
    return image_array, initial_image_copy

def graph_distance_by_iteration(distance_list):
    dist_array = np.array(distance_list)
    total_dists = np.sum(dist_array,axis=1)
    plt.plot(total_dists)

def random_inital_medoids(image_array,k):
    unique_vals,unique_index = np.unique(image_array,axis=0,return_index=True)
    medoid_indexes = np.random.choice(unique_index,size=k,replace=False)
    mean_array = image_array[medoid_indexes,:]
    return mean_array.astype(np.int16)

def manually_set_centroids(image_array,k,centroids_array):
    assert image_array.shape[1] == centroids_array.shape[1], "Shape of input centers don't match your image array shape"
    assert k == centroids_array.shape[0], "Shape of input centers don't match your number of k values required"
    return centroids_array

def reshape_plot(compressed_array,original_image):
    original_image_copy = np.copy(original_image)
    iheight,iwidth,ipx = original_image_copy.shape
    im_array = np.reshape(compressed_array,(iheight,iwidth,ipx))
    im_array = im_array.astype(np.int16)
    plt.imshow(im_array)
    return plt.show()

def labels_to_means(initial_image_array,means_array,labels):
    assigned_means = np.copy(initial_image_array)
    increment = 0
    for i in labels:
        assigned_means[increment,:] = means_array[i,:]
        increment += 1
    return assigned_means

def ecludian_distance(image_array,means_array,return_icd=False):
    image_array_float = np.copy(image_array).astype(np.float)
    means_array_float = np.copy(means_array).astype(np.float)
    expanded_image_array = np.repeat(image_array_float[:,:,np.newaxis],means_array_float.shape[0],axis=2)
    expanded_means = means_array_float[:,:,np.newaxis].T
    mean_labels = np.argmin(np.sum(np.square(expanded_image_array - expanded_means),axis=1),axis=1)
    if return_icd:
        inter_clust_dists = []
        for i in range(0,means_array_float.shape[0]):
            inter_cluster_index = np.argwhere(mean_labels == i)[:,0]
            inter_cluster_values = image_array_float[inter_cluster_index,:]
            current_inter_cluster_distance = np.sum(np.square(inter_cluster_values - means_array_float[i,:]))
            inter_clust_dists.append(current_inter_cluster_distance)
        return mean_labels,np.sum(inter_clust_dists)
    else:
        return mean_labels

def manhatten_distance(image_array,means_array,return_icd=False):
    image_array_float = np.copy(image_array).astype(np.float)
    means_array_float = np.copy(means_array).astype(np.float)
    expanded_image_array = np.repeat(image_array_float[:,:,np.newaxis],means_array_float.shape[0],axis=2)
    expanded_means = means_array_float[:,:,np.newaxis].T
    mean_labels = np.argmin(np.sum(np.absolute(expanded_image_array - expanded_means),axis=1),axis=1)
    if return_icd:
        inter_clust_dists = []
        for i in range(0,means_array_float.shape[0]):
            inter_cluster_index = np.argwhere(mean_labels == i)[:,0]
            inter_cluster_values = image_array_float[inter_cluster_index,:]
            current_inter_cluster_distance = np.sum(np.absolute(inter_cluster_values - means_array_float[i,:]))
            inter_clust_dists.append(current_inter_cluster_distance)
        return mean_labels,np.sum(inter_clust_dists)
    else:
        return mean_labels

def re_solve_k_means(image_array,means_array,labels):
    new_means = np.zeros((means_array.shape[0],means_array.shape[1]),dtype=float)
    for i in range(0,means_array.shape[0]):
        inter_cluster_index = np.argwhere(labels == i)[:,0]
        new_means[i,:] = np.average(image_array[inter_cluster_index,:],axis=0)
    
    return new_means.astype(np.int16)

def K_means(initial_image_path,k=3,iter_max=500,distance='ecludian',sweeping_k=False,man_input=False,input_array=None):
    if sweeping_k == False:
        initial_image_array, original_array = image_to_numpy(initial_image_path,show=True)
    else:
        initial_image_array, original_array = image_to_numpy(initial_image_path,show=False)
    iteration_labels = np.zeros((initial_image_array.shape[0]))
    iteration_mean = random_inital_medoids(initial_image_array,k=k)

    if man_input:
        iteration_mean = input_array
        assert input_array.shape[0] == k

    if distance == 'manhatten':
        iteration_labels = manhatten_distance(initial_image_array,iteration_mean)
    else:
        iteration_labels = ecludian_distance(initial_image_array,iteration_mean)

    if sweeping_k == False:
        print('------- Image Based on Random Centers and Assignment  -------')
        reshape_plot(labels_to_means(initial_image_array,iteration_mean,iteration_labels),original_array)

    iterations_count = 0
    for i in range(0,iter_max):
        iterations_count += 1
        new_centroids= re_solve_k_means(initial_image_array,iteration_mean,iteration_labels)
        if np.array_equal(new_centroids,iteration_mean):
            if sweeping_k == False:
                print('------The K-Means Algorithim has reached convergence halted during iteration %s ------' % i)
            break
        else:
            iteration_mean = new_centroids
            if distance == 'manhatten':
                iteration_labels = manhatten_distance(initial_image_array,iteration_mean)
            else:
                iteration_labels = ecludian_distance(initial_image_array,iteration_mean)

    if sweeping_k:
        final_centers = iteration_mean
        if distance == 'manhatten':
            centers_f,final_interation_distance = ecludian_distance(initial_image_array,final_centers,return_icd=True)
        else:
            centers_f,final_interation_distance = ecludian_distance(initial_image_array,final_centers,return_icd=True)
        return np.array(iteration_labels), final_interation_distance, final_centers, iterations_count

    else:    
        reshape_plot(labels_to_means(initial_image_array,iteration_mean,iteration_labels),original_array)
        return iteration_mean, np.array(iteration_labels)

def sweep_k_values(initial_image_path,iter_max=50,start_k=2,end_k=200,distance='ecludian',show_images=False,suppres_text=True):
    k_distance_list = []
    k_indexes = range(start_k,end_k)
    times_list = []
    number_of_iterations_required = []
    for i in range(start_k,end_k):
        start_time = time.time()
        final_labels,k_distance,final_centers,iterations_count  = K_means(initial_image_path,k=i,iter_max=iter_max,distance='ecludian',sweeping_k=True)
        end_time = time.time()
        times_list.append(end_time-start_time)
        number_of_iterations_required.append(iterations_count)
        k_distance_list.append(k_distance)
        if suppres_text == False:
            print('-------- K value %s converged--------'% i)
    return k_distance_list,times_list,k_indexes,number_of_iterations_required