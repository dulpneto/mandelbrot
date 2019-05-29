#include <png.h>

void setRGB(png_byte *ptr, int val, int iterations);
int writeImage(char* filename, int width, int height, int iterations, int *buffer);