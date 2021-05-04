//My (somewhat) optimized and debugged CDA code. This code will run a CDA, see the comments at the bottom for how you might call it and plot it.
//If you want to use PCA to get the initial embedding, then uncomment the import and the pca code/call.  I strongly recommend against this, as PCA was not helpful when I used it and the package is very finnicky.
//All of the commented out code after CDA is some examples of how you might use the function, including how to call the included python plotting files.
//These python files expect a certain file path where they can save their output, but the plotting is difficult to write generally so think of what I've written there as more insiration than hard tools
//I also recommend against using the neural gas. Its implementation is correct, but is quite touchy and most likely will hinder any attempt to use CDA. It is included for the sake of completeness and/or modification.
//The neural gas is also largely unnecessary as the performance boost from omp was enough to make the algorithm reasonable to use on large datasets (facespace took 30 minutes or so and it was 4096x700 or so).
//There is significant room to make the code more efficient: Transposing and vectorizing to x86 word length or perhaps even running it on a gpu would be good steps (in that order, the gpu is overkill)
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
#include <limits.h> 
#include <queue>
#include <functional>
#include "omp.h"
// #include "opencv2/core.hpp"
// #include "opencv2/imgproc.hpp"
// #include "opencv2/highgui.hpp"
#include "vector_operations.cpp"
#include <fstream>
#include <cfloat>



PyObject* convertToList(std::vector<std::vector<double>> *data){
    PyObject* result = PyList_New(0);
    for (int vec_index=0;vec_index<data->size();vec_index++){
        PyObject* vector = PyList_New(0);
        for (int dim=0;dim<(*data)[0].size();dim++){ 
            PyList_Append(vector, PyFloat_FromDouble((*data)[vec_index][dim]));
        }
        PyList_Append(result, vector);
    }
    return result;
}

PyObject* convertToList(std::vector<double> *data){
    PyObject* result = PyList_New(0);
    for (int vec_index=0;vec_index<data->size();vec_index++){
        PyList_Append(result, PyFloat_FromDouble((*data)[vec_index]));
    }
    return result;
}

double get_data_distance(std::vector<std::vector<double>> const *y_distances, int y1_index, int y2_index){
    return (*y_distances)[y1_index][y2_index];
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


double get_error(std::vector<std::vector<double>> const *data, std::vector<std::vector<double>> const *y_distances, std::vector<std::vector<double>> *embedded_data, double F(double, double), std::vector<double> lambda_values){
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
            double local_error = (first_term * F(dx, lambda_values[i]));
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
        for (int y2_index=0;y2_index<data->size();y2_index++){
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

std::vector<int> get_random_order(int length){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::vector<int> ordering(length);
    std::iota((ordering.begin()), (ordering.end()), 0);
    std::shuffle(ordering.begin(), ordering.end(), std::default_random_engine(seed));
    return ordering;
}

std::vector<std::vector<double>> random_initial(std::vector<std::vector<double>> const *data, std::vector<std::vector<double>> const *y_distances, int dim, int N){
    std::vector<std::vector<double>> embedded_data;
    float a = getMaxDataDistance(y_distances);
    for(int i=0; i<N;i++){
        std::vector<double> temp;
        for(int j=0; j<dim;j++){
            temp.push_back((((float)rand()/(float)(RAND_MAX)) * a));
        }
        embedded_data.push_back(temp);
    }
    return embedded_data;
}

std::vector<std::vector<double>> random_NG_initial(std::vector<std::vector<double>> const *data, int N){
    std::vector<std::vector<double>> embedded_data;
    int dim = (*data)[0].size();
    std::vector<int> indices = get_random_order(data->size());
    for (int index_index=0; index_index<N;index_index++){
        std::vector<double> temp;
        for(int j=0; j<dim;j++){
            temp.push_back((*data)[indices[index_index]][j]);
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
}



  //dijstra algorithm modified from: https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
int minDistance(std::vector<double> *dist, std::vector<bool> sptSet) 
{ 
    // Initialize min value 
    double min = DBL_MAX, min_index; 
  
    for (int v = 0; v < dist->size(); v++) 
        if (sptSet[v] == false && (*dist)[v] <= min) 
            min = (*dist)[v], min_index = v; 
  
    return min_index; 
} 

// Function that implements Dijkstra's single source shortest path algorithm 
// for a graph represented using adjacency matrix representation 
std::vector<double> dijkstra(std::vector<std::vector<double>> *graph, int src) 
{ 
    std::vector<double> dist(graph->size(), DBL_MAX); // The output array.  dist[i] will hold the shortest 
    // distance from src to i 
  
    std::vector<bool> sptSet(graph->size(), false); // sptSet[i] will be true if vertex i is included in shortest 
    // path tree or shortest distance from src to i is finalized 
  
    // Distance of source vertex from itself is always 0 
    dist[src] = 0; 
  
    // Find shortest path for all vertices 
    for (int count = 0; count < graph->size() - 1; count++) { 
        // Pick the minimum distance vertex from the set of vertices not 
        // yet processed. u is always equal to src in the first iteration. 
        int u = minDistance(&dist, sptSet); 
  
        // Mark the picked vertex as processed 
        sptSet[u] = true; 
  
        // Update dist value of the adjacent vertices of the picked vertex. 
        for (int v = 0; v < graph->size(); v++) 
  
            // Update dist[v] only if is not in sptSet, there is an edge from 
            // u to v, and total weight of path from src to  v through u is 
            // smaller than current value of dist[v] 
            if (!sptSet[v] && (*graph)[u][v]!=DBL_MAX && dist[u]!=DBL_MAX 
                && dist[u] + (*graph)[u][v] < dist[v]) 
                dist[v] = dist[u] + (*graph)[u][v]; 
    } 
  
    // print the constructed distance array 
    return dist; 
} 



void makeDiagonal(std::vector<std::vector<double>> *graphDistances){//graph distances must be square. We will make it symmetric and then diagonal
    for (int y1_index=0;y1_index<graphDistances->size();y1_index++){
        for (int y2_index=y1_index+1;y2_index<graphDistances->size();y2_index++){ 
            (*graphDistances)[y1_index][y2_index] = std::min((*graphDistances)[y1_index][y2_index], (*graphDistances)[y2_index][y1_index]);
        }
    }
    return;
}

std::vector<std::pair<double, int>> getKSmallest(std::vector<std::vector<double>> const *distances, int srcIndex, int K)
{
    std::priority_queue<std::pair<double, int>> pointDistances;
    for (int i = 0; i < (*distances).size(); i++) {
        pointDistances.push(std::pair<double, int>(-1*get_data_distance(distances, srcIndex, i), i));
    }
    std::vector<std::pair<double, int>> kclosest;
    for(int i = 0; i<K; i++){
        kclosest.push_back(std::pair<double, int>(-1*pointDistances.top().first, pointDistances.top().second));
        pointDistances.pop();
    }
    return kclosest;
}

std::vector<std::vector<double>> getKSimGraph(std::vector<std::vector<double>> const *y_distances, int K, bool weightEnabled){
    std::vector<std::vector<double>> KClosestNeighbordSimilarityGraph;
    for(int i=0; i<y_distances->size();i++){
        std::vector<std::pair<double, int>> kneighbors = getKSmallest(y_distances, i, K);
        std::vector<double> relativedistancesto_i(y_distances->size(), DBL_MAX);
        relativedistancesto_i[i] = 0;
        for(int j=0; j<K;j++){
            relativedistancesto_i[kneighbors[j].second] = weightEnabled ? kneighbors[j].first : 1;
        }
        KClosestNeighbordSimilarityGraph.push_back(relativedistancesto_i);
    }
    makeDiagonal(&KClosestNeighbordSimilarityGraph);
    for(int i=0; i<y_distances->size();i++){
        int num_edges=0;
        for(int j=0; j<y_distances->size();j++)
            if (KClosestNeighbordSimilarityGraph[i][j] < DBL_MAX){
                num_edges++;
            }
    }
    return KClosestNeighbordSimilarityGraph;
}

std::vector<std::vector<double>> geodesic_data_distances(std::vector<std::vector<double>> const *data, std::vector<std::vector<double>> K_rule_graph){
    std::vector<std::vector<double>> distances;
    for(int i=0; i<data->size(); i++){
        std::vector<double> y1_distances;
        auto dijkstraDistances = dijkstra(&K_rule_graph, i); 
        distances.push_back(dijkstraDistances);
    }
    makeDiagonal(&distances);
    return distances;
}


void updateLambda(std::vector<double> *lambdas, std::vector<std::vector<double>> *embedded_data, int x_index, int epoch, double desired_pi, double F(double, double)){
    std::vector<int> indicesOfPointsInNeighborhood;
    for(int j=0; j<lambdas->size(); j++){
        if(F(euclidean_distance(&((*embedded_data)[x_index]), &((*embedded_data)[j])), (*lambdas)[x_index]) == 1){
            indicesOfPointsInNeighborhood.push_back(j);
        }
    }
    float P = 2;
    (*lambdas)[x_index] = (*lambdas)[x_index] * pow((double) desired_pi/((double) indicesOfPointsInNeighborhood.size()/((double) lambdas->size())), 1.0/((double) P));
    
}

std::vector<double> getInitialLambdas(std::vector<std::vector<double>> *embedded_data){
    std::vector<double> lambda_values;
    for(int i = 0; i<embedded_data->size(); i++){
        std::vector<double> dx;
        for(int j = 0; j<embedded_data->size(); j++){
            dx.push_back(euclidean_distance(&((*embedded_data)[i]), &((*embedded_data)[j])));
        }
        double biggest = *max_element(dx.begin(), dx.end());
        lambda_values.push_back(biggest);
    }
    return lambda_values;
}
double excitation(double ki, double lambda){
    return exp(-ki/lambda);
}

std::vector<double> get_updated_prototype(std::vector<double> *w, std::vector<double> *v, double lambda, double epsilon, double Dv){
    double term1 = epsilon*excitation(Dv, lambda);
    std::vector<double> vector_diff = vector_subtraction(v, w);
    std::vector<double> multiplied = scalar_multiplication(term1, &vector_diff);
    std::vector<double> test = vector_addition(w, &multiplied);
    return test;
}


std::vector<int> getSortOrder(std::vector<double> *v, std::vector<std::vector<double>> *prototypes){
    std::vector<std::pair<double, int>> distance_and_index;
    for(int i = 0; i< (*prototypes).size(); i++){
        distance_and_index.push_back(std::pair<double, int>(euclidean_distance(v, &(*prototypes)[i]), i));
    }
    sort(distance_and_index.begin(), distance_and_index.end());
    std::vector<int> sortOrder;
    for(int i = 0; i< distance_and_index.size(); i++){
        sortOrder.push_back(distance_and_index[i].second);
    }
    return sortOrder;
}


std::vector<std::vector<double>> getDBLMAXMatrix(int num_centers){
    std::vector<std::vector<double>> toReturn(
    num_centers,
    std::vector<double>(num_centers, DBL_MAX));
    return toReturn;
}

std::vector<std::vector<int>> getZeroesMatrix(int num_centers){
    std::vector<std::vector<int>> toReturn(
    num_centers,
    std::vector<int>(num_centers, 0));
    return toReturn;
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> neuralGas(std::vector<std::vector<double>> const *data, int num_centers, int max_epochs, double epsilon_initial, double epsilon_final, double lambda_initial, double lambda_final, int T_initial, int T_final, bool are_edges_weighted){
    int dim = (*data)[0].size();

    std::vector<std::vector<double>> adjacencyGraph = getDBLMAXMatrix(num_centers);
    std::vector<std::vector<int>> ageMatrix = getZeroesMatrix(num_centers);
    std::vector<std::vector<double>> y_distances = data_distances(data);
    std::vector<std::vector<double>> prototypes = random_NG_initial(data, num_centers); //take random data points

    for(int epoch = 0; epoch<max_epochs; epoch++){
        printf("NG epoch: %d\n", epoch);
        std::vector<int> x1_indices = get_random_order(data->size());
        for (int x1_index_index=0; x1_index_index<x1_indices.size();x1_index_index++){
            std::vector<double> v = (*data)[x1_indices[x1_index_index]];

            std::vector<int> SortOrder = getSortOrder(&v, &prototypes);
            
            double epsilon = epsilon_initial*pow((epsilon_final/epsilon_initial),(epoch/max_epochs));
            double lambda = lambda_initial*pow((lambda_final/lambda_initial),(epoch/max_epochs));
            double T = T_initial*pow((T_final/T_initial),(epoch/max_epochs));

            for(int ki=0; ki<num_centers; ki++){
                int i=SortOrder[ki];
                prototypes[i] = get_updated_prototype(&prototypes[i], &v, lambda, epsilon, ki);
            }

            int first_winner_index = SortOrder[0];
            int second_winner_index = SortOrder[1];

            double matrixEntry = are_edges_weighted ? euclidean_distance(&prototypes[first_winner_index], &prototypes[second_winner_index]) : 1;
            adjacencyGraph[first_winner_index][second_winner_index] = matrixEntry;
            adjacencyGraph[second_winner_index][first_winner_index] = matrixEntry;

            ageMatrix[first_winner_index][second_winner_index] = 0;
            ageMatrix[second_winner_index][first_winner_index] = 0;

            for(int i = 0; i<num_centers; i++){
                ageMatrix[first_winner_index][i]++;
                ageMatrix[i][first_winner_index]++;
                if(ageMatrix[i][first_winner_index] > T){
                    adjacencyGraph[first_winner_index][i] = DBL_MAX;
                    adjacencyGraph[i][first_winner_index] = DBL_MAX;    
                }
            }
        }           
    }
    return std::make_tuple(prototypes, adjacencyGraph);
}


/*
does a CDA reduction, returns a vector of vectors, where the inner vectors are the data points, a vector of vectors that represents the computed similarity graph, a vector of error over the run, and a pointer to the data that was reduced
inputdata: the input data: vector of data points (also vectors)
F: the neighborhood function. is passed: dx (euclidean distance in embedding space), lambda
F': the derivative of the neighborhood function. is passed: dx, lambda
dim: The desired output dimension
alpha: The initial learning rate
alpha_update: The update function at each epoch for the learning rate. Gets passed(current alpha, final desired alpha, current epoch, max epoch)
quantize: indicates whether or not to use neural gas to both quantize the input data and produce the similarity graph. I RECOMMEND AGAINST THIS, ONLY USE IT IF YOU KNOW WHAT YOU ARE DOING
do_pca: disabled, meant to indicate whether to run pca for intiail embedding
alpha_n: Final value of alpha used by the sample exponential update functions 
int max_epoch: maximum number of iterations
int K: the number of closest neighbors to use for the K-closest neighbor similarity graph, does nothing if neural gas is enabled
float desired_pi: the proportion of data to keep in the neighborhood width of each point, see lee verleyson nonlinear dimensionality reduction for a specification on how this works

note: Since the neural gas was never tuned to my satisfaction I did not fully integrate it, its parameters must be hard coded in in line 491. Again only use this if you know what you are doing.
*/
std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<double>, std::vector<std::vector<double>>> cda(std::vector<std::vector<double>> const *inputdata, double F(double, double), double F_prime(double, double), int dim, double alpha, double alpha_update(double, double, int, int), bool quantize, bool do_pca, double alpha_n, int max_epoch, int K, double desired_pi){
    std::vector<std::vector<double>> K_rule_graph;
    std::vector<std::vector<double>> data_deref;
    std::vector<std::vector<double>> const *data;
    if (quantize){
        std::tie(data_deref, K_rule_graph) = neuralGas(inputdata, 150, 500, 0.8, 0.05, 4.7, .1, 30, 40, true);
        data = &data_deref;
    }
    else{
        data = inputdata;
        K=K+1;
        std::vector<std::vector<double>> euclidean_y_distances = data_distances(data);
        K_rule_graph = getKSimGraph(&euclidean_y_distances, K, true);

    }
    
    std::vector<std::vector<double>> y_distances = geodesic_data_distances(data, K_rule_graph);
    std::vector<std::vector<double>> embedded_data;
    printf("initializing random data ...");
    // if (do_pca){
    //     embedded_data = pca_initial(data, dim, &y_distances);
    // }
    // else{
        embedded_data = random_initial(data, &y_distances, dim, data->size());   
    // }
    printf("calculating initial lambda values ...");
    std::vector<double> lambda_values = getInitialLambdas(&embedded_data);

    std::list<double> error = {};
    printf("calculating initial error ...");
    error.push_back(get_error(data, &y_distances, &embedded_data, F, lambda_values));

    int epoch = 0;
    while(!hasConverged(&error) && epoch < max_epoch){
        printf("Epoch %d \n", epoch);

        alpha = alpha_update(alpha, alpha_n, epoch, max_epoch);

        std::vector<int> x1_indices = get_random_order(embedded_data.size());
        for (int x1_index_index=0; x1_index_index<x1_indices.size();x1_index_index++){
            int x1_index = x1_indices[x1_index_index];
            std::vector<int> x2_indices = get_random_order(embedded_data.size());
            #pragma omp parallel for
            for (int x2_index_index=0; x2_index_index<x2_indices.size();x2_index_index++){
                int x2_index = x2_indices[x2_index_index];
                if (!(x1_index == x2_index)){
                    auto updated_x2 = get_updated_x2(x1_index, x2_index, &embedded_data, &y_distances, alpha, lambda_values[x1_index], F, F_prime);
                    #pragma omp critical
                    embedded_data[x2_index] = updated_x2;
                }
            }
            updateLambda(&lambda_values, &embedded_data, x1_index, epoch, desired_pi, step_function);
        }
        double epoch_error = get_error(data, &y_distances, &embedded_data, F, lambda_values);
        error.push_back(epoch_error);
        printf("Error: %f \n", epoch_error);
        epoch+=1;
    }
    std::vector<double> error_vector{ std::begin(error), std::end(error) };
    return std::make_tuple(embedded_data, K_rule_graph, error_vector, *data);
}


// double decreasing_alpha(double alpha, double alpha_n, int epoch, int max_epoch){ //what can we use to satisfy the robbins monro conditions?
//     double test = alpha*pow((alpha_n/alpha), epoch/(max_epoch-1));
//     if(std::isnan(test)){
//         printf("alpha: %lf, alpha_n: %lf, epoch: %d, max_epoch: %d \n", alpha, alpha_n, epoch, max_epoch);
//     }
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

// std::vector<std::vector<double>> generateParabola(){
//     std::vector<std::vector<double>> data= {};
//     #pragma omp parallel for
//     for(int i=-3; i<4;i++){
//         for(int j=-1; j<3;j++){
//             std::vector<double> temp;
//             temp.push_back(i);
//             temp.push_back(pow(i,2));
//             temp.push_back(j*1);
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
//         temp.push_back(4*(cos(phi)));
//         temp.push_back(4*(sin(phi)));
//         temp.push_back(0);
//         data.push_back(temp);
//     }
//     return data;
// }

// std::vector<std::vector<double>> generate2DCircle(int num_points){
//     std::vector<std::vector<double>> data= {};
//     float a = 100.0;
//     for(int i=0; i<num_points;i++){
//         std::vector<double> temp;
//         double phi = (((float)rand()/(float)(RAND_MAX)) * a)-50;
//         temp.push_back(4*(cos(phi)));
//         temp.push_back(4*(sin(phi)));
//         data.push_back(temp);
//     }
//     return data;
// }

// std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<double>, std::vector<std::vector<double>>> test1(){
//     std::vector<std::vector<double>> data= generateWideDeterministicSwissRoll(80);
//     double alpha = .5;
//     std::vector<std::vector<double>> embedded_data;
//     std::vector<std::vector<double>> reduced_data;
//     std::vector<std::vector<double>> k_rule_graph;
//     std::vector<double> error;

//     std::tie(embedded_data, k_rule_graph, error, reduced_data) = cda(&data, step_function, d_step_function, 2, alpha, decreasing_alpha, false, false, .01, 300, 8, .6);
    
//     return std::make_tuple(data, embedded_data, k_rule_graph, error, reduced_data);
// }

// std::vector<std::vector<double>> readFaces(){
//     std::vector<std::vector<double>> data;
//     std::ifstream faceFile("faceMatrix.csv");
//     char * pch;

//     std::string line;
//     while(std::getline(faceFile, line)){
//         std::vector<double> result;

//         std::istringstream iss(line);
//         std::string token;
//         while (std::getline(iss, token, ',')){
//             result.push_back(stod(token));
//         }
//         data.push_back(result);
//     }
//     return data;
// }

// std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> matrix){
//     std::vector<std::vector<double>> toRet;
//     for(int i=0; i<matrix[0].size();i++){
//         std::vector<double> col;
//         toRet.push_back(col);
//     }
//     for(int i=0; i<matrix.size();i++){
//         for(int j=0; j<matrix[0].size();j++){
//             toRet[j].push_back(matrix[i][j]);
//         }
//     }
//     return toRet;
// }


// std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<double>, std::vector<std::vector<double>>> faces(){
//     printf("reading csv... \n");
//     std::vector<std::vector<double>> data = transpose(readFaces());
//     printf("completed reading csv \n");
//     double alpha = .5;
//     std::vector<std::vector<double>> embedded_data;
//     std::vector<std::vector<double>> reduced_data;
//     std::vector<std::vector<double>> k_rule_graph;
//     std::vector<double> error;

//     std::tie(embedded_data, k_rule_graph, error, reduced_data) = cda(&data, step_function, d_step_function, 2, alpha, decreasing_alpha, false, false, .01, 300, 8, .6);

//     return std::make_tuple(data, embedded_data, k_rule_graph, error, reduced_data);
// }


// int plotfaces(std::vector<std::vector<double>> cdata, std::vector<std::vector<double>> cembedded_data, std::vector<std::vector<double>> ck_rule_graph, std::vector<double> cerror, std::vector<std::vector<double>> creduced_data){
//     Py_Initialize();
//     PyRun_SimpleString("import sys");
//     PyRun_SimpleString("import os");
//     PyRun_SimpleString("sys.path.append(os.getcwd())");
//     PyObject* data = convertToList(&(cdata));
//     PyObject* embedded_data = convertToList(&(cembedded_data));
//     PyObject* reduced_data = convertToList(&(creduced_data));
//     PyObject* k_rule_graph = convertToList(&(ck_rule_graph));
//     PyObject* error = convertToList(&(cerror));
//     long type = 1;
//     PyObject* red_type = PyLong_FromLong(type);

//     PyObject *arglist;
    
    
//     PyObject* args = PyTuple_New(5);
//     if (PyTuple_SetItem(args, 0, reduced_data) != 0) {         
//         throw "python argument construction failed";
//     }

//     if (PyTuple_SetItem(args, 1, embedded_data) != 0) {         
//         throw "python argument construction failed";
//     }
//     if (PyTuple_SetItem(args, 2, red_type) != 0) {         
//         throw "python argument construction failed";
//     }
//     if (PyTuple_SetItem(args, 3, k_rule_graph) != 0) {         
//         throw "python argument construction failed";
//     }if (PyTuple_SetItem(args, 4, error) != 0) {         
//         throw "python argument construction failed";
//     }
//     auto module_name = PyUnicode_FromString("faceplotter");
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
//     if (PyErr_Occurred()) {
//     PyErr_PrintEx(0);
//     PyErr_Clear(); // this will reset the error indicator so you can run Python code again
// }
//     Py_DECREF(python_class);
//   } else {
//     std::cout << "Cannot instantiate the Python class" << std::endl;
//     Py_DECREF(python_class);
//     return 1;
//   }
//     Py_Finalize();
//     return 0;
// }

int main(){
//     std::vector<std::vector<double>> cdata;
//     std::vector<std::vector<double>> cembedded_data;
//     std::vector<std::vector<double>> creduced_data;
//     std::vector<std::vector<double>> ck_rule_graph;
//     std::vector<double> cerror;
//     std::tie(cdata, cembedded_data, ck_rule_graph, cerror, creduced_data) = faces();
//     plotfaces(cdata, cembedded_data, ck_rule_graph, cerror, creduced_data);
    return 0;
}

// int plot3d2s(std::vector<std::vector<double>> cdata, std::vector<std::vector<double>> cembedded_data, std::vector<std::vector<double>> ck_rule_graph, std::vector<double> cerror, std::vector<std::vector<double>> creduced_data){
//     Py_Initialize();
//     PyRun_SimpleString("import sys");
//     PyRun_SimpleString("import os");
//     PyRun_SimpleString("sys.path.append(os.getcwd())");
//     PyObject* data = convertToList(&(cdata));
//     PyObject* embedded_data = convertToList(&(cembedded_data));
//     PyObject* reduced_data = convertToList(&(creduced_data));
//     PyObject* k_rule_graph = convertToList(&(ck_rule_graph));
//     PyObject* error = convertToList(&(cerror));
//     long type = 1;
//     PyObject* red_type = PyLong_FromLong(type);

//     PyObject *arglist;
    
    
//     PyObject* args = PyTuple_New(5);
//     if (PyTuple_SetItem(args, 0, reduced_data) != 0) {         
//         throw "python argument construction failed";
//     }

//     if (PyTuple_SetItem(args, 1, embedded_data) != 0) {         
//         throw "python argument construction failed";
//     }
//     if (PyTuple_SetItem(args, 2, red_type) != 0) {         
//         throw "python argument construction failed";
//     }
//     if (PyTuple_SetItem(args, 3, k_rule_graph) != 0) {         
//         throw "python argument construction failed";
//     }if (PyTuple_SetItem(args, 4, error) != 0) {         
//         throw "python argument construction failed";
//     }
//     auto module_name = PyUnicode_FromString("similarity_plotter");
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
//     auto python_class = PyDict_GetItemString(dict, "plot_cda_output");
//     if (python_class == nullptr) {
//         PyErr_Print();
//         std::cerr << "Fails to get the Python class.\n";
//         return 1;
//     }
//   Py_DECREF(dict);
//   if (PyCallable_Check(python_class)) {
//     auto object = PyObject_CallObject(python_class, args);
//     if (PyErr_Occurred()) {
//     PyErr_PrintEx(0);
//     PyErr_Clear(); // this will reset the error indicator so you can run Python code again
// }
//     Py_DECREF(python_class);
//   } else {
//     std::cout << "Cannot instantiate the Python class" << std::endl;
//     Py_DECREF(python_class);
//     return 1;
//   }
//     Py_Finalize();
//     return 0;
// }