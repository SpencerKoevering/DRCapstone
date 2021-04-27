#A first pass python implementation of CCA. This code is far too slow to run on any large dataset. I also have not done any thorough debugging of it, so there may be errors with its output. Seriously, just use the c++ implementation

import numpy as np
from sklearn.decomposition import PCA
import random
import profile

#data: a list of n dimensional vectors
def cca(data, F, F_prime, lambda_value, lambda_update, dim, alpha, alpha_update, quantize):
    if quantize:
        data = VQ(data)

    y_distances = data_distances(data)
    embedded_data = pca_initial(data, dim)

    error = []
    error.append(get_error(data, y_distances, embedded_data, F, lambda_value))

    epoch = 0
    while(not hasConverged(error)):
        alpha = alpha_update(alpha, epoch)
        lambda_value = lambda_update(lambda_value, epoch)
        print("Running Epoch:", epoch)
        for x1_index in np.random.permutation(len(embedded_data)): #range(len(embedded_data)):
            for x2_index in np.random.permutation(len(embedded_data)):
                if not x1_index == x2_index:
                    embedded_data[x2_index] = get_updated_x2(x1_index, x2_index, embedded_data, y_distances, alpha, lambda_value, F, F_prime)
        error.append(get_error(data, y_distances, embedded_data, F, lambda_value))
        epoch+=1
    return embedded_data

def VQ(data):
    return 1

def hasConverged(error):
    if(len(error) <= 20):
        return False
    elif((abs(error[-1]-error[-2]) + abs(error[-2]-error[-3]) + abs(error[-3]-error[-4]) + abs(error[-4]-error[-5])) < .01):
        return True
    else:
        return False

def get_error(data, y_distances, embedded_data, F, lambda_value):
    total_error=0
    for i in range(len(data)):
        for j in range(len(data)):
            if(i==j):
                continue
            dx = euclidean_distance(embedded_data[i], embedded_data[j])
            total_error+=((y_distances[i][j]-dx)**2 * F(dx, lambda_value))
    return total_error

def data_distances(data):
    distances = {}
    for y1_index in range(len(data)):
        if not y1_index in distances.keys():
            distances[y1_index] = {}
        for y2_index in range(len(data)):
            if y1_index == y2_index:
                continue
            else:
                distances[y1_index][y2_index] = np.linalg.norm(data[y1_index]-data[y2_index])
    return distances


def pca_initial(data, dim):
    pca = PCA(n_components = dim)
    embedded_data = pca.fit_transform(data)
    return embedded_data

def random_order(length):
    random.shuffle(list(range(length)))

def beta(x1_index, x2_index, embedded_data, y_distances, F, F_prime, lambda_value):
    d_y = y_distances[x1_index][x2_index]
    d_x =  euclidean_distance(embedded_data[x1_index], embedded_data[x2_index])
    # if not (2*F(d_x, lambda_value) > (d_y-d_x)*F_prime(d_x, lambda_value)):
    #     raise Exception("Violated condition on Beta") 
    return (d_y - d_x)*(2*F(d_x, lambda_value) - (d_y - d_x)*F_prime(d_x, lambda_value))

def get_updated_x2(x1_index, x2_index, embedded_data, y_distances, alpha, lambda_value, F, F_prime):
    return embedded_data[x2_index]-alpha*beta(x1_index, x2_index, embedded_data, y_distances, F, F_prime, lambda_value)*((embedded_data[x1_index]-embedded_data[x2_index])/(euclidean_distance(embedded_data[x1_index], embedded_data[x2_index])))
    #see equation 4.82 in lee verleyson

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1-p2)








def step_function(Yij, lambda_value): #what is the derivative???
    if (Yij <= lambda_value):
        return 1
    else:
        return 0

def d_step_function(Yij, lambda_value):
    return 0

def decreasing_lambda(lambda_value, epoch):
    return lambda_value*.9

def decreasing_alpha(alpha, epoch): #what can we use to satisfy the robbins monro conditions?
    return .9*alpha

def main():
    data = []
    for i in range(50):
        data.append(np.array([random.randint(-50,50), random.randint(-50,50), random.randint(-50,50)]))
    alpha = .5

    distances = data_distances(data)
    max_distance = 0

    for x1_dict in distances.values():
        for x1x2Distance in x1_dict.values():
            if(x1x2Distance > max_distance):
                max_distance = x1x2Distance

    lambda_value = .9*max_distance


    reduced = cca(data, step_function, d_step_function, lambda_value, decreasing_lambda, 2, alpha, decreasing_alpha, False)
    print(reduced)

main()