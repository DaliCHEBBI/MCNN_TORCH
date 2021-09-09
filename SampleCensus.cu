#include <stdio.h>
#include <assert.h>
#include <math_constants.h>
#include <stdint.h>
#include <unistd.h>
#include <png++/image.hpp>
#include <torch/torch.h>
#include "SampleCensus.cuh"

/***********************************************************************/
__device__ void sort(float *x, int n)
{
	for (int i = 0; i < n - 1; i++) {
		int min = i;
		for (int j = i + 1; j < n; j++) {
			if (x[j] < x[min]) {
				min = j;
			}
		}
		float tmp = x[min];
		x[min] = x[i];
		x[i] = tmp;
	}
}



__device__ void csort(float *x, int n)
{
 sort(x,n);
 }
