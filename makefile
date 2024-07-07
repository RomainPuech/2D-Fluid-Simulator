CC = g++
CFLAGS = -O3 -fopenmp -fdiagnostics-color=always -g

all: main

main: main.cpp
	$(CC) $(CFLAGS) -o main main.cpp

clean:
	rm -f main