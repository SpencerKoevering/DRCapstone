//My (somewhat) optimized and debugged CCA code. This code will run a CCA, see the comments at the bottom for how you might call it and plot it.
//If you want to use PCA to get the initial embedding, then uncomment the import and the pca code/call.  I recommend against this, as PCA was not helpful when I used it.
//All of the commented out code after cca is some examples of how you might use the function, including how to call the enclosed python plotting files. 
//These python files expect a certain file path where they can save their output, but the plotting is difficult to write generally so think of what I've written there as more insiration than hard tools 
// Please email vankoesd@whitman.edu with questions
#include <Python.h> 
#include <stdio.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <list>
#include <chrono>
#include <random>
#include <algorithm>
#include <math.h>      
#include <execution>
#include "omp.h"
// #include "opencv2/core.hpp"
// #include "opencv2/imgproc.hpp"
// #include "opencv2/highgui.hpp"
#include "vector_operations.cpp"

double get_data_distance(std::vector<std::vector<double>> const *y_distances, int y1_index, int y2_index){
    double toReturn;
    if(y1_index == y2_index){
        return 0;
    }
    else if(y1_index > y2_index){
        toReturn = (*y_distances)[y2_index][y1_index-y2_index-1]; //remember that distances are symmetric, so we just need the larger index second to be in the diagonal matrix
    }
    else{
        toReturn = (*y_distances)[y1_index][y2_index-y1_index-1];
    }
    if (toReturn == -1){
        printf("Data distance index out of range: (%d,%d)\n",y1_index, y2_index);
        throw -1;
    }
    return toReturn;
}

bool hasConverged(std::list<double> const *error_pointer){
    std::list<double> error = *error_pointer;
    if(error.size() <= 20){
        return false;
    }

    double cumulative_error = 0;
    
    std::list<double>::iterator current = error.end();
    --current;
    double previous = *current;
    for (int i=0; i<5; i++)
    {
        --current; 
        cumulative_error+=std::abs(*current-previous);
        previous=*current;
    }
    if(cumulative_error < .00000000001){
        return true;
    }
    else{
        return false;
    }
}


double get_error(std::vector<std::vector<double>> const *data, std::vector<std::vector<double>> const *y_distances, std::vector<std::vector<double>> *embedded_data, double F(double, double), double lambda_value){
    double total_error=0;
    #pragma omp parallel for
    for (int i=0;i<data->size();i++){
        // #pragma omp parallel for
        for (int j=i+1;j<data->size();j++){
            if(i==j){
                continue;
            }
            double dx = euclidean_distance(&((*embedded_data)[i]), &((*embedded_data)[j]));
            double first_term = pow((get_data_distance(y_distances, i, j)-dx), 2);
            double local_error = (first_term * F(dx, lambda_value));
            #pragma omp critical
            total_error = total_error + local_error;
        }
    }
    return .5*total_error;
}

std::vector<std::vector<double>> data_distances(std::vector<std::vector<double>> const *data){
    std::vector<std::vector<double>> y_distances;
    for (int y1_index=0;y1_index<data->size();y1_index++){
        std::vector<double> y1_distances;
        for (int y2_index=y1_index+1;y2_index<data->size();y2_index++){ //distances are symmetric so we only need a diagonal matrix
            y1_distances.push_back(euclidean_distance(&((*data)[y1_index]), &((*data)[y2_index])));
        }
        y_distances.push_back(y1_distances);
    }
    return y_distances;
}


// std::vector<std::vector<double>> pca_initial(std::vector<std::vector<double>> const *data, int dim, std::vector<std::vector<double>> const *y_distances){
//     cv::Mat dataMatrix;
//     for(int data_dim=0;data_dim<((*data)[0]).size();data_dim++){
//         std::vector<double> dimI_array;
//         for(int vector_index=0;vector_index<data->size();vector_index++){
//             dimI_array.push_back((*data)[vector_index][data_dim]);
//         }
//         cv::Mat dimI_matrix(dimI_array,true);
//         dimI_matrix=dimI_matrix.t();
//         dataMatrix.push_back(dimI_matrix);
//     }
//     cv::PCA pca(dataMatrix, // pass the data
//         cv::Mat(), // we do not have a pre-computed mean vector,
//                // so let the PCA engine to compute it
//         cv::PCA::DATA_AS_COL, // indicate that the vectors
//                             // are stored as matrix rows
//                             // (use PCA::DATA_AS_COL if the vectors are
//                             // the matrix columns)
//         dim // specify, how many principal components to retain
//     );
//     std::vector<std::vector<double>> embedded_data;
//     double sample_distance = get_data_distance(y_distances, 0, 1);
//     for(int vector_index=0;vector_index<data->size();vector_index++){
//         auto projected = pca.project(dataMatrix.col(vector_index));
//         std::vector<double> projected_vector;
//         projected.col(0).copyTo(projected_vector);
//         #pragma omp parallel for
//         for(int j=0; j<vector_index-1; j++){
//             if(projected_vector[0] == embedded_data[j][0] && projected_vector[1] == embedded_data[j][1]){
//                 #pragma omp critical
//                 projected_vector[0] += static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/sample_distance));
//                 #pragma omp critical
//                 projected_vector[1] += static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/sample_distance));
//             }
//         }
//         embedded_data.push_back(projected_vector); 
//     }
//     return embedded_data;
// }


double getMaxDataDistance(std::vector<std::vector<double>> const *y_distances){
    double max = 0;
    for(int i=0;i<(*y_distances).size(); i++){
        for(int j=0;j<(*y_distances)[i].size(); j++){
            max = ((*y_distances)[i][j] > max) ? (*y_distances)[i][j] : max; 
        }       
    }
    return max;
}

std::vector<std::vector<double>> random_initial(std::vector<std::vector<double>> const *data, std::vector<std::vector<double>> const *y_distances, int dim){
    std::vector<std::vector<double>> embedded_data;
    float a = getMaxDataDistance(y_distances);
    for(int i=0; i<data->size();i++){
        std::vector<double> temp;
        for(int j=0; j<dim;j++){
            temp.push_back((((float)rand()/(float)(RAND_MAX)) * a));
        }
        embedded_data.push_back(temp);
    }
    return embedded_data;
}

double beta(int x1_index, int x2_index, std::vector<std::vector<double>> const *embedded_data, std::vector<std::vector<double>> const *y_distances, double F(double, double), double F_prime(double, double), double lambda_value, double x1x2euclideandistance){
    int first_data_index = x2_index;
    int second_data_index = x1_index;
    if(x1_index> x2_index){
        first_data_index = x1_index;
        second_data_index = x2_index;
    }
    double dycheck = (*y_distances)[second_data_index][first_data_index];
    double d_y = get_data_distance(y_distances, x1_index, x2_index);
    double d_x = x1x2euclideandistance;
    // if not (2*F(d_x, lambda_value) > (d_y-d_x)*F_prime(d_x, lambda_value)){
    //     raise Exception("Violated condition on Beta") 
    //}
    double Fvalue = F(d_x, lambda_value);
    double F_prime_value = F_prime(d_x, lambda_value);
    double test = (d_y - d_x)*(2*Fvalue - (d_y - d_x)*F_prime_value);
    return test;
}

std::vector<double> get_updated_x2(int x1_index, int x2_index, std::vector<std::vector<double>> const *embedded_data, std::vector<std::vector<double>> const *y_distances, double alpha, double lambda_value, double F(double, double), double F_prime(double, double)){
    std::vector<double> x1x2difference = vector_subtraction(&((*embedded_data)[x1_index]), &(*embedded_data)[x2_index]);
    double x1x2euclideandistance = euclidean_distance(&(*embedded_data)[x1_index], &(*embedded_data)[x2_index]);
    
    std::vector<double> term1 = scalar_division(&x1x2difference, x1x2euclideandistance);
    double beta_term = beta(x1_index, x2_index, embedded_data, y_distances, F, F_prime, lambda_value, x1x2euclideandistance);
    std::vector<double> rawupdatevector = scalar_multiplication(beta_term, &term1);
    std::vector<double> deltax2 = scalar_multiplication(alpha, &rawupdatevector);
    std::vector<double> test = vector_subtraction(&((*embedded_data)[x2_index]), &deltax2);
    return vector_subtraction(&((*embedded_data)[x2_index]), &deltax2);
    //#see equation 4.82 in lee verleyson
}


std::vector<int> get_random_order(int length){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::vector<int> ordering(length);
    std::iota((ordering.begin()), (ordering.end()), 0);
    std::shuffle(ordering.begin(), ordering.end(), std::default_random_engine(seed));
    return ordering;
}

/*
does a CCA reduction, returns a vector of vectors, where the inner vectors are the data points
Data: the input data: vector of data points (also vectors)
F: the neighborhood function. is passed: lambda
F': the derivative of the neighborhood function. is passed: lambda
lambda_value: The initial value of the neighborhood width
lambda_update: The function that updates lambda at each timestep (current lambda, final desired lambda, current epoch, max epoch)
dim: The desired output dimension
alpha: The initial learning rate
alpha_update: The update function at each epoch for the learning rate. Gets passed(current alpha, final desired alpha, current epoch, max epoch)
quantize: disabled, meant to indicate whether to run vector quantization
do_pca: disabled, meant to indicate whether to run pca for intiail embedding
alpha_n: Final value of alpha used by the sample exponential update functions 
double lambda_n: Final value of lambda used by the sample exponential update functions
 int max_epoch: maximum number of iterations
*/
std::vector<std::vector<double>> cca(std::vector<std::vector<double>> const *data, double F(double, double), double F_prime(double, double), double lambda_value, double lambda_update(double, double, int, int), int dim, double alpha, double alpha_update(double, double, int, int), bool quantize, bool do_pca, double alpha_n, double lambda_n, int max_epoch){

    std::vector<std::vector<double>> y_distances = data_distances(data);
    std::vector<std::vector<double>> embedded_data;
    // if (do_pca){
    //     embedded_data = pca_initial(data, dim, &y_distances);
    // }
    // else{
        embedded_data = random_initial(data, &y_distances, dim);   
    // }
    std::list<double> error = {};
    error.push_back(get_error(data, &y_distances, &embedded_data, F, lambda_value));

    int epoch = 0;
    while(!hasConverged(&error) && epoch < max_epoch){
        printf("Epoch %d \n", epoch);

        alpha = alpha_update(alpha, alpha_n, epoch, max_epoch);
        lambda_value = lambda_update(lambda_value, lambda_n, epoch, max_epoch);
        std::vector<int> x1_indices = get_random_order(embedded_data.size());
        for (int x1_index_index=0; x1_index_index<x1_indices.size();x1_index_index++){
            int x1_index = x1_indices[x1_index_index];
            std::vector<int> x2_indices = get_random_order(embedded_data.size());
            #pragma omp parallel for
            for (int x2_index_index=0; x2_index_index<x2_indices.size();x2_index_index++){
                int x2_index = x2_indices[x2_index_index];
                if (!(x1_index == x2_index)){
                    auto updated_x2 = get_updated_x2(x1_index, x2_index, &embedded_data, &y_distances, alpha, lambda_value, F, F_prime);
                    #pragma omp critical
                    embedded_data[x2_index] = updated_x2;
                }
            }
        }
        double epoch_error = get_error(data, &y_distances, &embedded_data, F, lambda_value);
        error.push_back(epoch_error);
        printf("Error: %f \n", epoch_error);
        epoch+=1;
    }
    printf("lambda = %lf, alpha = %lf \n", lambda_value, alpha);
    return embedded_data;
}



// double step_function(double Yij, double lambda_value){
//     if (Yij <= lambda_value){
//         return 1;
//     }
//     else{
//         return 0;
//     }
// }

// double d_step_function(double Yij, double lambda_value){
//     return 0;
// }


// double decreasing_lambda(double lambda_value, double lambda_n, int epoch, int max_epoch){
//     return lambda_value* pow((lambda_n/lambda_value), epoch/(max_epoch-2));
// }

// double decreasing_alpha(double alpha, double alpha_n, int epoch, int max_epoch){ //what can we use to satisfy the robbins monro conditions?
//     double test = alpha*pow((alpha_n/alpha), epoch/(max_epoch-2));
//     return test;
// }


// std::vector<std::vector<double>> generateSwissRoll(int num_points){
//     std::vector<std::vector<double>> data= {};
//     float a = 20;
//     #pragma omp parallel for
//     for(int i=0; i<num_points;i++){
//         std::vector<double> temp;
//         double phi = (((float)rand()/(float)(RAND_MAX)) * a)+ 4.5;
//         temp.push_back(phi*(cos(phi)));
//         temp.push_back(phi*(sin(phi)));
//         temp.push_back((((float)rand()/(float)(RAND_MAX)) * a)-10);
//         #pragma omp critical
//         data.push_back(temp);
//     }
//     return data;
// }

// std::vector<std::vector<double>> generateDeterministicSwissRoll(int num_points){
//     std::vector<std::vector<double>> data= {};
//     #pragma omp parallel for
//     for(int i=0; i<num_points;i++){
//         double phi = (.1*i+1);
//         for(int j=-1; j<3;j++){
//             std::vector<double> temp;
//             temp.push_back(10*phi*(cos(phi)));
//             temp.push_back(10*phi*(sin(phi)));
//             temp.push_back(j*2*5);
//             #pragma omp critical
//             data.push_back(temp);
//         }
//     }
//     return data;
// }

// std::vector<std::vector<double>> generateThinDeterministicSwissRoll(int num_points){
//     std::vector<std::vector<double>> data= {};
//     #pragma omp parallel for
//     for(int i=0; i<num_points;i++){
//         double phi = (.1*i+1);
//         for(int j=-1; j<3;j++){
//             std::vector<double> temp;
//             temp.push_back(10*phi*(cos(phi)));
//             temp.push_back(10*phi*(sin(phi)));
//             temp.push_back(j*1);
//             #pragma omp critical
//             data.push_back(temp);
//         }
//     }
//     return data;
// }

// std::vector<std::vector<double>> generateWideDeterministicSwissRoll(int num_points){
//     std::vector<std::vector<double>> data= {};
//     #pragma omp parallel for
//     for(int i=0; i<num_points;i++){
//         double phi = (.1*i+1);
//         for(int j=-1; j<3;j++){
//             std::vector<double> temp;
//             temp.push_back(10*phi*(cos(phi)));
//             temp.push_back(10*phi*(sin(phi)));
//             temp.push_back(j*8);
//             #pragma omp critical
//             data.push_back(temp);
//         }
//     }
//     return data;
// }

// std::vector<std::vector<double>> generateCircle(int num_points){
//     std::vector<std::vector<double>> data= {};
//     float a = 100.0;
//     for(int i=0; i<num_points;i++){
//         std::vector<double> temp;
//         double phi = (((float)rand()/(float)(RAND_MAX)) * a)-50;
//         temp.push_back((cos(phi)));
//         temp.push_back((sin(phi)));
//         temp.push_back(0);
//         data.push_back(temp);
//     }
//     return data;
// }

// std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> test1(){
//     std::vector<std::vector<double>> data= generateWideDeterministicSwissRoll(80);
//     double alpha = .6;

//     std::vector<std::vector<double>> distances = data_distances(&data);
//     double max_distance = 0;


//     for (int x1_index=0;x1_index<distances.size();x1_index++){
//         for (int x1x2Distance_index=0; x1x2Distance_index < distances[x1_index].size(); x1x2Distance_index++){
//             if(distances[x1_index][x1x2Distance_index] > max_distance){
//                 max_distance = get_data_distance(&distances, x1_index, x1x2Distance_index);
//             }
//         }
//     }

//     double lambda_value = 400;

//     std::vector<std::vector<double>> embedded_data = cca(&data, step_function, d_step_function, lambda_value, decreasing_lambda, int(2), alpha, decreasing_alpha, false, false, .01, 50, 1000);
    

//     return std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>(data, embedded_data);
// }

// PyObject* convertToList(std::vector<std::vector<double>> *data){
//     PyObject* result = PyList_New(0);
//     for (int vec_index=0;vec_index<data->size();vec_index++){
//         PyObject* vector = PyList_New(0);
//         for (int dim=0;dim<(*data)[0].size();dim++){ 
//             PyList_Append(vector, PyFloat_FromDouble((*data)[vec_index][dim]));
//         }
//         PyList_Append(result, vector);
//     }
//     return result;
// }



int main(){
//     std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> reduction = test1();
//     Py_Initialize();
//     PyRun_SimpleString("import sys");
//     PyRun_SimpleString("sys.path.append(\".\")");
    
//     PyObject* data = convertToList(&(reduction.first));
//     PyObject* embedded_data = convertToList(&(reduction.second));
//     long type = 0;
//     PyObject* red_type = PyLong_FromLong(type);

//     PyObject *arglist;
    
    
//     PyObject* args = PyTuple_New(3);
//     if (PyTuple_SetItem(args, 0, data) != 0) {         
//         throw "python argument construction failed";
//     }

//     if (PyTuple_SetItem(args, 1, embedded_data) != 0) {         
//         throw "python argument construction failed";
//     }
//     if (PyTuple_SetItem(args, 2, red_type) != 0) {         
//         throw "python argument construction failed";
//     }
//     auto module_name = PyUnicode_FromString("cca_plotter");
//     auto module = PyImport_Import(module_name);
//     if (module == nullptr) {
//         PyErr_Print();
//         std::cerr << "Fails to import the module.\n";
//         return 1;
//     }
//     Py_DECREF(module_name);
//     auto dict = PyModule_GetDict(module);
//     if (dict == nullptr) {
//         PyErr_Print();
//         std::cerr << "Fails to get the dictionary.\n";
//         return 1;
//     }
//     Py_DECREF(module);

//     // Builds the name of a callable class
//     auto python_class = PyDict_GetItemString(dict, "plot_dim_reduction");
//     if (python_class == nullptr) {
//         PyErr_Print();
//         std::cerr << "Fails to get the Python class.\n";
//         return 1;
//     }
//   Py_DECREF(dict);
//   if (PyCallable_Check(python_class)) {
//     auto object = PyObject_CallObject(python_class, args);
//     Py_DECREF(python_class);
//   } else {
//     std::cout << "Cannot instantiate the Python class" << std::endl;
//     Py_DECREF(python_class);
//     return 1;
//   }

//     Py_Finalize();
    return 0;
}