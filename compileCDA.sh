FLAGS="$(pkg-config --cflags --libs opencv4)"
g++ cda.cpp -fopenmp $FLAGS -I/usr/include/python3.8/ -lpython3.8 -o cda.out 
