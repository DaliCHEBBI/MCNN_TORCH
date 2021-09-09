#pragma once
#include <stdio.h>
#include <assert.h>
#include <math_constants.h>
#include <stdint.h>
#include <unistd.h>
#include <png++/image.hpp>
#include <torch/torch.h>

//__device__ void sort(float *x, int n);

void csort(float *x, int n);
