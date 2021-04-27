FLAGS="$(pkg-config --cflags --libs opencv4)"
g++ cca.cpp -fopenmp $FLAGS -I/usr/include/python3.8/ -lpython3.8 -o cca.out
