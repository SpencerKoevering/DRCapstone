//This file contains some basic vector operations used by both cca and cda 

double euclidean_distance(std::vector<double> const *p1, std::vector<double> const *p2){
    double sum = 0;
    for(int i=0;i<p1->size();i++){
        #pragma omp critical
        sum+= pow((*p1)[i]-(*p2)[i], 2);
    }
    #pragma omp critical
    sum = sqrt(sum);
    return sum;
}

std::vector<double> vector_subtraction(std::vector<double> const *p1, std::vector<double> const *p2){
    std::vector<double> p3 = {};
    for(int i=0;i<p1->size();i++){
        #pragma omp critical
        p3.push_back((*p1)[i]-(*p2)[i]);
    }
    return p3;
}

std::vector<double> vector_addition(std::vector<double> const *p1, std::vector<double> const *p2){
    std::vector<double> p3 = {};
    for(int i=0;i<p1->size();i++){
        #pragma omp critical
        p3.push_back((*p1)[i]+(*p2)[i]);
    }
    return p3;
}

std::vector<double> scalar_division(std::vector<double> const *p1, double scalar){
    std::vector<double> p2 = {};
    for(int i=0;i<p1->size();i++){
        #pragma omp critical
        p2.push_back((*p1)[i]/scalar);
    }
    return p2;
}

std::vector<double> scalar_multiplication(double scalar, std::vector<double> const *p1){ //switched the order here to jive nicely with the notation of the update rule in lee verlyeson
    std::vector<double> p2 = {};
    for(int i=0;i<p1->size();i++){
        p2.push_back((*p1)[i]*scalar);
    }
    return p2;
}
