#include <stdio.h>
#include <assert.h>
#include <math_constants.h>
#include <stdint.h>
#include <unistd.h>
#include <png++/image.hpp>
#include <torch/torch.h>
#include "Census.cuh"

#define DISP_MAX 256


#define COLOR_DIFF(x, i, j) (abs(x[i] - x[j]))

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
/***********************************************************************/
__global__ void ad(float *x0, float *x1, float *output, int size, int size2, int size3, int direction)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size) {
		int d = id;
		int x = d % size3;
		d /= size3;
		int y = d % size2;
		d /= size2;
		d *= direction;

		float dist;
		if (0 <= x + d && x + d < size3) {
			int cnt = 0;
			dist = 0;
			for (int yy = y - 4; yy <= y + 4; yy++) {
				for (int xx = x - 4; xx <= x + 4; xx++) {
					if (0 <= xx && xx < size3 && 0 <= xx + d && xx + d < size3 && 0 <= yy && yy < size2) {
						int ind = yy * size3 + xx;
						dist += abs(x0[ind] - x1[ind + d]);
						cnt++;
					}
				}
			}
			dist /= cnt;
		} else {
			dist = CUDART_NAN;
		}
		output[id] = dist;
	}
}
/***********************************************************************/
__global__ void census(float *x0, float *x1, float *output, int size, int num_channels, int size2, int size3, int direction)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size) {
		int d = id;
		int x = d % size3;
		d /= size3;
		int y = d % size2;
		d /= size2;
		d *= direction;

		float dist;
		if (0 <= x + d && x + d < size3) {
			dist = 0;
			for (int i = 0; i < num_channels; i++) {
				int ind_p = (i * size2 + y) * size3 + x;
				for (int yy = y - 4; yy <= y + 4; yy++) {
					for (int xx = x - 4; xx <= x + 4; xx++) {
						if (0 <= xx && xx < size3 && 0 <= xx + d && xx + d < size3 && 0 <= yy && yy < size2) {
							int ind_q = (i * size2 + yy) * size3 + xx;
							if ((x0[ind_q] < x0[ind_p]) != (x1[ind_q + d] < x1[ind_p + d])) {
								dist++;
							}
						} else {
							dist++;
						}
					}
				}
			}
			dist /= num_channels;
		} else {
			dist = CUDART_NAN;
		}
		output[id] = dist;
	}
}
/***********************************************************************/
#if 0
__global__ void add_vol(float *vol, float *cnt, float *out, int size, int size1, int size2, int size3, float ratio)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int d = id;
		int x = d % size3;
		d /= size3;
		int y = d % size2;
		d /= size2;

		int lo = floor(d * ratio);
		int hi = lo + 1;
		float alpha = (d * ratio) - lo;
		assert(0 <= lo && hi < size1);

		float val = vol[(lo * size2 + y) * size3 + x] * (1 - alpha) + vol[(hi * size2 + y) * size3 + x] * alpha;
		if (!isnan(val) && cnt[id] > 0) {
			out[id] += val;
			cnt[id] += 1;
		}
	}
}

__global__ void rho(float *x, int size, float lambda)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		x[id] = 1 - exp(-x[id] / lambda);
	}
}

#endif

__global__ void spatial_argmin(float *input, float *output, int size, int size1, int size23)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int dim23 = id % size23;
		int dim0 = id / size23;

		int argmin = 0;
		float min = CUDART_INF;
		for (int i = 0; i < size1; i++) {
			float val = input[(dim0 * size1 + i) * size23 + dim23];
			if (val < min) {
				min = val;
				argmin = i;
			}
		}
		output[id] = argmin + 1;
	}
}
/***********************************************************************/
__global__ void cross(float *x0, float *out, int size, int dim2, int dim3, int L1, float tau1)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int dir = id;
		int x = dir % dim3;
		dir /= dim3;
		int y = dir % dim2;
		dir /= dim2;

		int dx = 0;
		int dy = 0;
		if (dir == 0) {
			dx = -1;
		} else if (dir == 1) {
			dx = 1;
		} else if (dir == 2) {
			dy = -1;
		} else if (dir == 3) {
			dy = 1;
		} else {
			assert(0);
		}

		int xx, yy, ind1, ind2, dist;
		ind1 = y * dim3 + x;
		for (xx = x + dx, yy = y + dy;;xx += dx, yy += dy) {
			if (xx < 0 || xx >= dim3 || yy < 0 || yy >= dim2) break;

			dist = max(abs(xx - x), abs(yy - y));
			if (dist == 1) continue;

			ind2 = yy * dim3 + xx;

			/* rule 1 */
			if (COLOR_DIFF(x0, ind1, ind2) >= tau1) break;

			/* rule 2 */
			if (dist >= L1) break;
		}
		out[id] = dir <= 1 ? xx : yy;
	}
}
/***********************************************************************/
__global__ void cbca(float *x0c, float *x1c, float *vol, float *out, int size, int dim2, int dim3, int direction)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int d = id;
		int x = d % dim3;
		d /= dim3;
		int y = d % dim2;
		d /= dim2;

		if (x + d * direction < 0 || x + d * direction >= dim3) {
			out[id] = vol[id];
		} else {
			float sum = 0;
			int cnt = 0;

			int yy_s = max(x0c[(2 * dim2 + y) * dim3 + x], x1c[(2 * dim2 + y) * dim3 + x + d * direction]);
			int yy_t = min(x0c[(3 * dim2 + y) * dim3 + x], x1c[(3 * dim2 + y) * dim3 + x + d * direction]);
			for (int yy = yy_s + 1; yy < yy_t; yy++) {
				int xx_s = max(x0c[(0 * dim2 + yy) * dim3 + x], x1c[(0 * dim2 + yy) * dim3 + x + d * direction] - d * direction);
				int xx_t = min(x0c[(1 * dim2 + yy) * dim3 + x], x1c[(1 * dim2 + yy) * dim3 + x + d * direction] - d * direction);
				for (int xx = xx_s + 1; xx < xx_t; xx++) {
					float val = vol[(d * dim2 + yy) * dim3 + xx];
					assert(!isnan(val));
					sum += val;
					cnt++;
				}
			}

			assert(cnt > 0);
			out[id] = sum / cnt;
			assert(!isnan(out[id]));
		}
	}
}
/***********************************************************************/
__global__ void sgm(float *x0, float *x1, float *vol, float *tmp, float *out, int dim1, int dim2, int dim3, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int sgm_direction, int direction)
{
	int x, y, dx, dy;

	dx = dy = 0;
	if (sgm_direction <= 1) {
		y = blockIdx.x * blockDim.x + threadIdx.x;
		if (y >= dim2) {
			return;
		}
		if (sgm_direction == 0) {
			x = 0;
			dx = 1;
		} else if (sgm_direction == 1) {
			x = dim3 - 1;
			dx = -1;
		}
	} else if (sgm_direction <= 3) {
		x = blockIdx.x * blockDim.x + threadIdx.x;
		if (x >= dim3) {
			return;
		}
		if (sgm_direction == 2) {
			y = 0;
			dy = 1;
		} else if (sgm_direction == 3) {
			y = dim2 - 1;
			dy = -1;
		}
	}

	assert(dim1 <= 400);
	float tmp_curr_[400];
	float tmp_prev_[400];
	float *tmp_curr = tmp_curr_;
	float *tmp_prev = tmp_prev_;

	float min_prev = CUDART_INF;
	for (; 0 <= y && y < dim2 && 0 <= x && x < dim3; x += dx, y += dy) {
		float min_curr = CUDART_INF;
		for (int d = 0; d < dim1; d++) {
			int ind = (d * dim2 + y) * dim3 + x;

			if (x + d * direction < 0 ||
				x + d * direction >= dim3 || 
				y - dy < 0 || 
				y - dy >= dim2 || 
				x + d * direction - dx < 0 || 
				x + d * direction - dx >= dim3 ||
				x - dx < 0 ||
				x - dx >= dim3) {

				out[ind] += vol[ind];
				tmp_curr[d] = vol[ind];
			} else {
				int ind2 = y * dim3 + x;

				float D1 = COLOR_DIFF(x0, ind2, ind2 - dy * dim3 - dx);
				float D2 = COLOR_DIFF(x1, ind2 + d * direction, ind2 + d * direction - dy * dim3 - dx);
				float P1, P2;
				if (D1 < tau_so && D2 < tau_so) { 
					P1 = pi1; 
					P2 = (pi1 * pi2); 
				} else if (D1 > tau_so && D2 > tau_so) { 
					P1 = pi1 / (sgm_q1 * sgm_q2);
					P2 = (pi1 * pi2) / (sgm_q1 * sgm_q2);
				} else {
					P1 = pi1 / sgm_q1;
					P2 = (pi1 * pi2) / sgm_q1;
				}

				assert(min_prev != CUDART_INF);
				float cost = min(tmp_prev[d], min_prev + P2);
				if (d > 0) {
					cost = min(cost, tmp_prev[d - 1] + (sgm_direction == 2 ? P1 / alpha1 : P1));
				}
				if (d < dim1 - 1) {
					cost = min(cost, tmp_prev[d + 1] + (sgm_direction == 3 ? P1 / alpha1 : P1));
				}
				float val = vol[ind] + cost - min_prev;
				out[ind] += val;
				tmp_curr[d] = val;
			}
			if (tmp_curr[d] < min_curr) {
				min_curr = tmp_curr[d];
			}
		}
		min_prev = min_curr;

		float *swap = tmp_curr;
		tmp_curr = tmp_prev;
		tmp_prev = swap;
	}
}
/***********************************************************************/
#define INDEX(dim0, dim1, dim2, dim3) \
	assert((dim1) >= 0 && (dim1) < size1 && (dim2) >= 0 && (dim2) < size2 && (dim3) >= 0 && (dim3) < size3), \
	((((dim0) * size1 + (dim1)) * size2 + (dim2)) * size3 + dim3)
/***********************************************************************/
template <int sgm_direction> __global__ void sgm2(float *x0, float *x1, float *input, float *output, float *tmp, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction, int size1, int size2, int size3, int step)
{
	int x, y, dx, dy;
	int d = threadIdx.x;

	if (sgm_direction == 0) {
		/* right */
		x = step;
		y = blockIdx.x;
		dx = 1;
		dy = 0;
	} else if (sgm_direction == 1) {
		/* left */
		x = size2 - 1 - step;
		y = blockIdx.x;
		dx = -1;
		dy = 0;
	} else if (sgm_direction == 2) {
		/* down */
		x = blockIdx.x;
		y = step;
		dx = 0;
		dy = 1;
	} else if (sgm_direction == 3) {
		/* up */
		x = blockIdx.x;
		y = size1 - 1 - step;
		dx = 0;
		dy = -1;
	}

	if (y - dy < 0 || y - dy >= size1 || x - dx < 0 || x - dx >= size2) {
		float val = input[INDEX(0, y, x, d)];
		output[INDEX(0, y, x, d)] += val;
		tmp[d * size2 + blockIdx.x] = val;
		return;
	}

	__shared__ float output_s[400], output_min[400];

	output_s[d] = output_min[d] = tmp[d * size2 + blockIdx.x];
	__syncthreads();

	for (int i = 256; i > 0; i /= 2) {
		if (d < i && d + i < size3 && output_min[d + i] < output_min[d]) {
			output_min[d] = output_min[d + i];
		}
		__syncthreads();
	}

	int ind2 = y * size2 + x;
	float D1 = COLOR_DIFF(x0, ind2, ind2 - dy * size2 - dx);
	float D2;
	int xx = x + d * direction;
	if (xx < 0 || xx >= size2 || xx - dx < 0 || xx - dx >= size2) {
		D2 = 10;
	} else {
		D2 = COLOR_DIFF(x1, ind2 + d * direction, ind2 + d * direction - dy * size2 - dx);
	}
	float P1, P2;
	if (D1 < tau_so && D2 < tau_so) {
		P1 = pi1;
		P2 = pi2;
	} else if (D1 > tau_so && D2 > tau_so) {
		P1 = pi1 / (sgm_q1 * sgm_q2);
		P2 = pi2 / (sgm_q1 * sgm_q2);
	} else {
		P1 = pi1 / sgm_q1;
		P2 = pi2 / sgm_q1;
	}

	float cost = min(output_s[d], output_min[0] + P2);
	if (d - 1 >= 0) {
		cost = min(cost, output_s[d - 1] + (sgm_direction == 2 ? P1 / alpha1 : P1));
	}
	if (d + 1 < size3) {
		cost = min(cost, output_s[d + 1] + (sgm_direction == 3 ? P1 / alpha1 : P1));
	}

	float val = input[INDEX(0, y, x, d)] + cost - output_min[0];
	output[INDEX(0, y, x, d)] += val;
	tmp[d * size2 + blockIdx.x] = val;
}
/***********************************************************************/

template <int sgm_direction> __global__ void sgm3(float *x0, float *x1, float *input, float *output, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction, int size1, int size2, int size3, int step)
{
	int x, y, dx, dy;
	int d = threadIdx.x;

	if (sgm_direction == 0) {
		/* right */
		x = step;
		y = blockIdx.x;
		dx = 1;
		dy = 0;
	} else if (sgm_direction == 1) {
		/* left */
		x = size2 - 1 - step;
		y = blockIdx.x;
		dx = -1;
		dy = 0;
	} else if (sgm_direction == 2) {
		/* down */
		x = blockIdx.x;
		y = step;
		dx = 0;
		dy = 1;
	} else if (sgm_direction == 3) {
		/* up */
		x = blockIdx.x;
		y = size1 - 1 - step;
		dx = 0;
		dy = -1;
	}

	if (y - dy < 0 || y - dy >= size1 || x - dx < 0 || x - dx >= size2) {
		output[INDEX(sgm_direction, y, x, d)] = input[INDEX(0, y, x, d)];
		return;
	}

	__shared__ float output_s[400], output_min[400];

	output_s[d] = output_min[d] = output[INDEX(sgm_direction, y - dy, x - dx, d)];
	__syncthreads();

	for (int i = 256; i > 0; i /= 2) {
		if (d < i && d + i < size3 && output_min[d + i] < output_min[d]) {
			output_min[d] = output_min[d + i];
		}
		__syncthreads();
	}

	int ind2 = y * size2 + x;
	float D1 = COLOR_DIFF(x0, ind2, ind2 - dy * size2 - dx);
	float D2;
	int xx = x + d * direction;
	if (xx < 0 || xx >= size2 || xx - dx < 0 || xx - dx >= size2) {
		D2 = 10;
	} else {
		D2 = COLOR_DIFF(x1, ind2 + d * direction, ind2 + d * direction - dy * size2 - dx);
	}
	float P1, P2;
	if (D1 < tau_so && D2 < tau_so) {
		P1 = pi1;
		P2 = pi2;
	} else if (D1 > tau_so && D2 > tau_so) {
		P1 = pi1 / (sgm_q1 * sgm_q2);
		P2 = pi2 / (sgm_q1 * sgm_q2);
	} else {
		P1 = pi1 / sgm_q1;
		P2 = pi2 / sgm_q1;
	}

	float cost = min(output_s[d], output_min[0] + P2);
	if (d - 1 >= 0) {
		cost = min(cost, output_s[d - 1] + (sgm_direction == 2 ? P1 / alpha1 : P1));
	}
	if (d + 1 < size3) {
		cost = min(cost, output_s[d + 1] + (sgm_direction == 3 ? P1 / alpha1 : P1));
	}

	output[INDEX(sgm_direction, y, x, d)] = input[INDEX(0, y, x, d)] + cost - output_min[0];
}

/***********************************************************************/
__global__ void fliplr(float *in, float *out, int size, int dim3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		out[id + dim3 - 2 * x - 1] = in[id];
	}
}
/***********************************************************************/

__global__ void outlier_detection(float *d0, float *d1, float *outlier, int size, int dim3, int disp_max)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int d0i = d0[id];
		if (x - d0i < 0) {
			//assert(0);
			outlier[id] = 1;
		} else if (abs(d0[id] - d1[id - d0i]) < 1.1) {
			outlier[id] = 0; /* match */
		} else {
			outlier[id] = 1; /* occlusion */
			for (int d = 0; d < disp_max; d++) {
				if (x - d >= 0 && abs(d - d1[id - d]) < 1.1) {
					outlier[id] = 2; /* mismatch */
					break;
				}
			}
		}
	}
}

/***********************************************************************/
#if 0

__global__ void iterative_region_voting(float *d0, float *x0c, float *x1c, float *outlier, float *d0_out, float *outlier_out, int size, int dim2, int dim3, float tau_s, float tau_h, int disp_max)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int y = id / dim3;
		
		d0_out[id] = d0[id];
		outlier_out[id] = outlier[id];

		if (outlier[id] == 0) return;

		assert(disp_max < DISP_MAX);
		int hist[DISP_MAX];
		for (int i = 0; i < disp_max; i++) {
			hist[i] = 0;
		}

		int yy_s = x0c[(2 * dim2 + y) * dim3 + x];
		int yy_t = x0c[(3 * dim2 + y) * dim3 + x];
		for (int yy = yy_s + 1; yy < yy_t; yy++) {
			int xx_s = x0c[(0 * dim2 + yy) * dim3 + x];
			int xx_t = x0c[(1 * dim2 + yy) * dim3 + x];
			for (int xx = xx_s + 1; xx < xx_t; xx++) {
				if (outlier[yy * dim3 + xx] == 0) {
					hist[(int)d0[yy * dim3 + xx]]++;
				}
			}
		}

		int cnt = 0;
		int max_i = 0;
		for (int i = 0; i < disp_max; i++) {
			cnt += hist[i];
			if (hist[i] > hist[max_i]) {
				max_i = i;
			}
		}

		if (cnt > tau_s && (float)hist[max_i] / cnt > tau_h) {
			outlier_out[id] = 0;
			d0_out[id] = max_i;
		}
	}
}


#endif

/***********************************************************************/
__global__ void interpolate_mismatch(float *d0, float *outlier, float *out, int size, int dim2, int dim3)
{
	const float dir[] = {
		0	,  1,
		-0.5,  1,
		-1	,  1,
		-1	,  0.5,
		-1	,  0,
		-1	, -0.5,
		-1	, -1,
		-0.5, -1,
		0	, -1,
		0.5 , -1,
		1	, -1,
		1	, -0.5,
		1	,  0,
		1	,  0.5,
		1	,  1,
		0.5 ,  1
	};

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		if (outlier[id] != 2) {
			out[id] = d0[id];
			return;
		}

		float vals[16];
		int vals_size = 0;

		int x = id % dim3;
		int y = id / dim3;
		for (int d = 0; d < 16; d++) {
			float dx = dir[2 * d];
			float dy = dir[2 * d + 1];
			float xx = x;
			float yy = y;
			int xx_i = round(xx);
			int yy_i = round(yy);
			while (0 <= yy_i && yy_i < dim2 && 0 <= xx_i && xx_i < dim3 && outlier[yy_i * dim3 + xx_i] == 2) {
				xx += dx;
				yy += dy;
				xx_i = round(xx);
				yy_i = round(yy);
			}

			int ind = yy_i * dim3 + xx_i;
			if (0 <= yy_i && yy_i < dim2 && 0 <= xx_i && xx_i < dim3) {
				assert(outlier[ind] != 2);
				vals[vals_size++] = d0[ind];
			}
		}
		assert(vals_size > 0);
		sort(vals, vals_size);
		out[id] = vals[vals_size / 2];
	}
}


__global__ void interpolate_occlusion(float *d0, float *outlier, float *out, int size, int dim3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		if (outlier[id] != 1) {
			out[id] = d0[id];
			return;
		}
		int x = id % dim3;

		int dx = 0;
		while (x + dx >= 0 && outlier[id + dx] != 0) {
			dx--;
		}
		if (x + dx < 0) {
			dx = 0;
			while (x + dx < dim3 && outlier[id + dx] != 0) {
				dx++;
			}
		}
		if (x + dx < dim3) {
			out[id] = d0[id + dx];
		} else {
			out[id] = d0[id];
		}
	}
}


#if 0

__global__ void sobel(float *x, float *g1, float *g2, int size, int dim2, int dim3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int xx = id % dim3;
		int yy = id / dim3;

		if (1 <= yy && yy < dim2 - 1 && 1 <= xx && xx < dim3 - 1) {
			g1[id] = -x[id-dim3-1] +x[id-dim3+1] -2*x[id-1] +2*x[id+1] -x[id+dim3-1] +x[id+dim3+1];
			g2[id] = x[id-dim3-1] +2*x[id-dim3] +x[id-dim3+1] -x[id+dim3-1] -2*x[id+dim3] -x[id+dim3+1];
		} else {
			g1[id] = 0;
			g2[id] = 0;
		}
	}
}


__global__ void depth_discontinuity_adjustment(float *d0, float *dg1, float *dg2, float *xg1, float *xg2, float *out, int size, int dim3, float tau_e)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		if (abs(dg1[id]) > tau_e) {
			out[id] = xg1[id - 1] > xg1[id + 1] ? d0[id - 1] : d0[id + 1];
		} else if (abs(dg2[id]) > tau_e) {
			out[id] = xg2[id - dim3] > xg2[id + dim3] ? d0[id - dim3] : d0[id + dim3];
		} else {
			out[id] = d0[id];
		}
	}
}

#endif

__global__ void subpixel_enchancement(float *d0, float *c2, float *out, int size, int dim23, int disp_max) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int d = d0[id];
		out[id] = d;
		if (1 <= d && d < disp_max - 1) {
			float cn = c2[(d - 1) * dim23 + id];
			float cz = c2[d * dim23 + id];
			float cp = c2[(d + 1) * dim23 + id];
			float denom = 2 * (cp + cn - 2 * cz);
			if (denom > 1e-5) {
				out[id] = d - min(1.0, max(-1.0, (cp - cn) / denom));
			}
		}
	}
}

__global__ void mean2d(float *img, float *kernel, float *out, int size, int kernel_radius, int dim2, int dim3, float alpha2)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int y = id / dim3;

		float sum = 0;
		float cnt = 0;
		int i = 0;
		for (int xx = x - kernel_radius; xx <= x + kernel_radius; xx++) {
			for (int yy = y - kernel_radius; yy <= y + kernel_radius; yy++, i++) {
				if (0 <= xx && xx < dim3 && 0 <= yy && yy < dim2 && abs(img[yy * dim3 + xx] - img[y * dim3 + x]) < alpha2) {
					sum += img[yy * dim3 + xx] * kernel[i];
					cnt += kernel[i];
				}
			}
		}
		out[id] = sum / cnt;
	}
}


__global__ void Normalize_get_norm_(float *input, float *norm, int size1, int size23, int size023)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size023) {
		int dim23 = id % size23;
		int dim0 = id / size23;

		float sum = 0.0;
		for (int dim1 = 0; dim1 < size1; dim1++) {
			float x = input[(dim0 * size1 + dim1) * size23 + dim23];
			sum += x * x;
		}
		norm[dim0 * size23 + dim23] = sum + 1e-5;
	}
}

__global__ void Normalize_forward_(float *input, float *norm, float *output, int size23, int size123, int size0123)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size0123) { 
		int dim23 = id % size23;
		int dim0 = (id / size123);
		output[id] = input[id] / sqrtf(norm[dim0 * size23 + dim23]);
	}
}


__global__ void Normalize_backward_input_(float *grad_output, float *input, float *norm, float *grad_input, int size1, int size23, int size0123)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size0123) {
		int dim0 = id;
		int dim23 = dim0 % size23;
		dim0 /= size23;
		int dim1 = dim0 % size1;
		dim0 /= size1;

		float denom = powf(norm[dim0 * size23 + dim23], 1.5);
		float deriv = (norm[dim0 * size23 + dim23] - input[id] * input[id]) / denom * grad_output[id];

		float sum = 0;
		for (int dim1_ = 0; dim1_ < size1; dim1_++) {
			if (dim1_ != dim1) {
				int ind = (dim0 * size1 + dim1_) * size23 + dim23;
				sum += input[ind] * grad_output[ind];
			}
		}
		grad_input[id] = deriv - sum * input[id] / denom;
	}
}


struct Margin2_functor {
	float margin;
	__host__ Margin2_functor(float margin_) : margin(margin_) {};
	__device__ float forward(float pos, float neg) {
		return fmaxf(0, neg - pos + margin);
	}
	__device__ float backward(float pos, float neg, int which) {
		float f = neg - pos + margin;
		if (which == 0) {
			return -1. * (f > 0);
		} else {
			return f > 0;
		}
	}
};

struct Margin2_squared_functor {
	float margin;
	__host__ Margin2_squared_functor(float margin_) : margin(margin_) {};
	__device__ float forward(float pos, float neg) {
		float d = fmaxf(0, neg - pos + margin);
		return d * d * 0.5;
	}
	__device__ float backward(float pos, float neg, int which) {
		float f = neg - pos + margin;
		if (which == 0) {
			return -f * (f > 0);
		} else {
			return f * (f > 0);
		}
	}
};

template <class Op> __global__ void Margin2_(float *input, float *tmp, float *gradInput, float margin, Op op, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		float pos = input[id * 2];
		float neg = input[id * 2 + 1];
		tmp[id] = op.forward(pos, neg);
		gradInput[id * 2] = op.backward(pos, neg, 0);
		gradInput[id * 2 + 1] = op.backward(pos, neg, 1);
	}
}


__global__ void StereoJoin_alternative_(float *input_L, float *input_R, float *output_L, float *output_R, int size1_input, int size1, int size3, int size23)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size23) {
		int dim3 = id % size3;
		assert(size1_input <= 128);
		float L_cache[128];
		for (int i = 0; i < size1_input; i++) {
			L_cache[i] = input_L[i * size23 + id];
		}

		for (int d = 0; d < size1; d++) {
			if (dim3 - d >= 0) {
				float sum = 0;
				for (int i = 0; i < size1_input; i++) {
					sum -= L_cache[i] * input_R[i * size23 + id - d];
				}
				output_L[d * size23 + id] = sum;
				output_R[d * size23 + id - d] = sum;
			}
		}
	}
}


__global__ void StereoL2R_(float *vol_L, float *vol_R, int size2, int size3, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int dim3 = id % size3;
		int dim1 = id / (size2 * size3);

		if (dim3 + dim1 >= size3) {
			vol_R[id] = CUDART_INF;
		} else {
			vol_R[id] = vol_L[id + dim1];
		}
	}
}


__global__ void bilateral_filter(float *img, float *out, int size, int dim2, int dim3, int kernel_radius, float sigma1, float sigma2)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int y = id / dim3;

		float sum = 0;
		float cnt = 0;
		for (int i = -kernel_radius; i <= kernel_radius; i++) {
			for (int j = -kernel_radius; j <= kernel_radius; j++) {
				int yy = y + i;
				int xx = x + j;
				if (0 <= xx && xx < dim3 && 0 <= yy && yy < dim2) {
					float color_diff = img[yy * dim3 + xx] - img[y * dim3 + x];
					float v1 = exp(-(i * i + j * j) / (2 * sigma1 * sigma1));
					float v2 = exp(-(color_diff * color_diff) / (2 * sigma2 * sigma2));
					sum += img[yy * dim3 + xx] * v1 * v2;
					cnt += v1 * v2;
				}
			}
		}
		out[id] = sum / cnt;
	}
}



__global__ void median2d(float *img, float *out, int size, int dim2, int dim3, int kernel_radius)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int y = id / dim3;

		float xs[11 * 11];
		int xs_size = 0;
		for (int xx = x - kernel_radius; xx <= x + kernel_radius; xx++) {
			for (int yy = y - kernel_radius; yy <= y + kernel_radius; yy++) {
				if (0 <= xx && xx < dim3 && 0 <= yy && yy < dim2) {
					xs[xs_size++] = img[yy * dim3 + xx];
				}
			}
		}
		sort(xs, xs_size);
		out[id] = xs[xs_size / 2];
	}
}


/***********************************************************************/
void readPNG16(torch::Tensor *imgT, const char * fname)   // See later how to make it a Float Tensor 
{
	//THFloatTensor *img_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
	//const char* fname = luaL_checkstring(L, 2);

	float *img = imgT->data_ptr<float>();
	png::image<png::gray_pixel_16> image(fname);
	int width = image.get_width();
	int height = image.get_height();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uint16_t val = image.get_pixel(j, i);
			img[i * width + j] = val == 0 ? 0.0 : ((float)val)/256.0;
		}
	}
}
/*******************************************************************/
/*******************************************************************/
void readPNGIARPA(torch::Tensor *imgT, const char * fname)
{
	//THFloatTensor *img_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
	//const char* fname = luaL_checkstring(L, 2);

	float *img = imgT->data_ptr<float>();
	png::image<png::gray_pixel_16> image(fname);
	int width = image.get_width();
	int height = image.get_height();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uint16_t val = image.get_pixel(j, i);
			img[i * width + j] = val == 0 ? 0.0 : ((float)val)/64.0;
		}
	}
}
/*******************************************************************/
/*******************************************************************/
void writePNG16(torch::Tensor imgT, int height, int width, const char * fname)
{
	float *img = imgT.data_ptr<float>();		
	png::image<png::gray_pixel_16> image(width, height);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float val = img[i * width + j];			
			image.set_pixel(j, i, (uint16_t)(val < 1e-5 ? 0 : val * 256.0));
		}
	}
	image.write(fname);
}
/*******************************************************************/

/*******************************************************************/
void writePFM(torch::Tensor imgT, const char * fname)
{
	//THFloatTensor *img_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
	//const char* fname = luaL_checkstring(L, 2);

	int height = imgT.size(0);    // size along dimension !!!!!!!!!!!!!!!!!!!
	int width = imgT.size(1);     // size along dimension !!!!!!!!!!!!!!!!!!!

	FILE *f = fopen(fname, "w");
	fprintf(f, "Pf\n%d %d\n-0.003922\n", width, height);
	fwrite(imgT.data_ptr<float>(), 4, height * width, f);
	fclose(f);
}
/*******************************************************************/
__global__ void remove_nonvisible(float *y, int size, int size3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % size3;
		if (y[id] >= x) {
			y[id] = 0;
		}
	}
}


__global__ void remove_occluded(float *y, int size, int size3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % size3;
		for (int i = 1; x + i < size3; i++) {
			if (i - y[id + i] < -y[id]) {
				y[id] = 0;
				break;
			}
		}
	}
}


__global__ void remove_white(float *x, float *y, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		if (x[id] == 255) {
			y[id] = 0;
		}
	}
}



__global__ void copy_fill(float *in, float *out, int size, int in_size2, int in_size3, int out_size2, int out_size3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int out_x = id % out_size3;
		int out_y = id / out_size3;

		int in_x = out_x - (out_size3 - in_size3) / 2;
		int in_y = out_y - (out_size2 - in_size2) / 2;

		int x = min(in_size3 - 1, max(0, in_x));
		int y = min(in_size2 - 1, max(0, in_y));

		out[id] = in[y * in_size3 + x];
	}
}



void memcpy2d(float *dst, float *src, int x, int y, int win_radius, int height, int width)
{
	assert(0 <= x - win_radius);
	assert(x + win_radius <= width);
	assert(0 <= y - win_radius);
	assert(y + win_radius <= height);
	for (int i = -win_radius; i <= win_radius; i++) {
		memcpy(dst, src + (y + i) * width + x - win_radius, (win_radius * 2 + 1) * sizeof(float));
		dst += win_radius * 2 + 1;
	}
}

double random_uniform() 
{
	return ((double)rand()/(double)RAND_MAX);
}

int random_int(int a, int b)
{
	assert(a <= b);
	return floor(random_uniform() * (b - a + 1) + a);
}

double random_exp(double lambda) 
{
	double u = random_uniform();
	return -log(u) / lambda;
}


/*******************************************************************/
void subset_dataset(torch::Tensor indexT, torch::Tensor inputT, torch::Tensor outputT )
{
	long *index = indexT.data_ptr<long>();                        //  check later !!!!!!!!
	float *input = inputT.data_ptr<float>();                       //  check later !!!!!!!!
	float *output = outputT.data_ptr<float>();                     //  check later !!!!!!!!

	const int N = 200;

	int set[N];
	for (int i = 0; i < N; i++) {
		set[i] = 0;
	}

	for (int i = 0; i < indexT.numel(); i++) {   // use of numel for the total number of elements
		assert(index[i] < N);
		set[index[i]] = 1;
	}

	int i = 0;
	for (int j = 0; j < inputT.size(0); j++) {
		int im = input[j * 4];
		if (set[im]) {
			for (int k = 0; k < 4; k++) {
				output[i * 4 + k] = input[j * 4 + k];
			}
			i++;
		}
	}
}

/*******************************************************************/
void make_dataset2(torch::Tensor dispT, torch::Tensor nnzT, int img, int t)
{
	//THFloatTensor *disp_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
	//THFloatTensor *nnz_ = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
	//int img = luaL_checkinteger(L, 3);
	//int t = luaL_checkinteger(L, 4);

	float *disp = dispT.data_ptr<float>();
	float *nnz  = nnzT.data_ptr<float>();

	int height = dispT.size(2);
	int width =  dispT.size(3);
	int nnz_size = nnzT.numel();

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (disp[i * width + j] > 0.5) {
				assert(t * 4 + 4 <= nnz_size);
				nnz[t * 4 + 0] = img;
				nnz[t * 4 + 1] = i;
				nnz[t * 4 + 2] = j;
				nnz[t * 4 + 3] = disp[i * width + j];
				t++;
			}
		}
	}
}

/* CPU implementation */
void grey2jet(torch::Tensor *grey_img, torch::Tensor *col_img)
{
	//THDoubleTensor *grey_img = (THDoubleTensor*)luaT_checkudata(L, 1, "torch.DoubleTensor");
	//THDoubleTensor *col_img = (THDoubleTensor*)luaT_checkudata(L, 2, "torch.DoubleTensor");

	assert(grey_img->ndimension == 2);
	if (3 * grey_img->numel() != col_img->numel()) {
		std::cerr << "Size mismatch\n";
	}

	int height = grey_img->size(0);
	int width =  grey_img->size(1);

	double *gray_data = grey_img->data_ptr<double>();
	double *col_data  = col_img->data_ptr<double>();

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			double val = gray_data[i * width + j] * 4;
			double r = 0, g = 0, b = 0;

			if (-0.1 <= val && val < 0.5) {
				r = 0;
				g = 0;
				b = 0.5 + val;
			} else if (0.5 <= val && val < 1.5) {
				r = 0;
				g = val - 0.5;
				b = 1;
			} else if (1.5 <= val && val < 2.5) {
				r = val - 1.5;
				g = 1;
				b = 1 - (val - 1.5);
			} else if (2.5 <= val && val < 3.5) {
				r = 1;
				g = 1 - (val - 2.5);
				b = 0;
			} else if (3.5 <= val && val <= 4.1) {
				r = 1 - (val - 3.5);
				g = 0;
				b = 0;
			} else {
				printf("val = %f\n", val);
				assert(0);
			}

			col_data[(0 * height + i) * width + j] = r;
			col_data[(1 * height + i) * width + j] = g;
			col_data[(2 * height + i) * width + j] = b;
		}
	}
}
