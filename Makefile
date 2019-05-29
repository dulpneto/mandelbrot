CC=nvcc
CCFLAGS=-O0 -Xcompiler -fopenmp -lpng
LDFLAGS=-lm -lpthread

all: mandelbrot

mandelbrot: mandelbrot.o write.o
	$(CC) $(CCFLAGS) -o $@ $^

mandelbrot.o: mandelbrot.cu write.h
	$(CC) $(CCFLAGS) -c $<

write.o: write.c write.h
	$(CC) $(CCFLAGS) -c $<

.PHONY: clean
clean:
	rm -f *.o mandelbrot
