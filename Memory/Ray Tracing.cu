//ʹ�ó����ڴ����ĳЩȫ���ڴ棬�������
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../common/cpu_bitmap.h"

#define rnc(x) (x * rand() / RAND_MAX)
#define SPHERES 20
#define INF 2e9
#define DIM 1024

struct Sphere
{
	float r, g, b;
	float x, y, z;
	float radius;
	
	__device__  float hit(int ox, int oy, float* n)
	{
		float dx = ox - x;
		float dy = oy - y;

		float t = sqrtf(dx * dx + dy * dy);
		if (t < radius)
		{
			float dz = sqrtf(radius * radius - t * t);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}

		return -INF;
	}
};

//���ó����ڴ�
__constant__ Sphere s[SPHERES];


__global__ void kernel(unsigned char* ptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int ox = x - DIM / 2;
	int oy = y - DIM / 2;

	float r = 0, g = 0, b = 0;
	float maxz = -INF;
	for (int i = 0; i < SPHERES; i++)
	{
		float n;
		float t = s[i].hit(ox, oy, &n);
		if (t > maxz)
		{
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;

			maxz = t;
		}
	}

	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;
}



int main()
{
	//�����¼����ڼ�ʱ
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;

	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

	//������ʱ��������ʼ��������GPUֵ
	Sphere* temp = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	for (int i = 0; i < SPHERES; i++)
	{
		temp[i].r = rnc(1.0f);
		temp[i].g = rnc(1.0f);
		temp[i].b = rnc(1.0f);
		temp[i].x = rnc(1000.0f) - 500;
		temp[i].y = rnc(1000.0f) - 500;
		temp[i].z = rnc(1000.0f) - 500;
		temp[i].radius = rnc(100.0f) + 20;
  	}
	cudaMemcpyToSymbol(s, temp, sizeof(Sphere) * SPHERES);
	free(temp);

	//�涨�̸߳��С
	dim3 grid(DIM / 16, DIM / 16);
	dim3 block(16, 16);

	kernel << <grid, block >> > (dev_bitmap);

	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

	//ֹͣ��ʱ
	cudaEventRecord(stop, 0);
	//ʱ��ͬ�����ȵ�stopǰ����䶼ִ���꣬Ȼ���ִ�н����������
	cudaEventSynchronize(stop);

	//����ʱ��
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Time to generate = %3.1f ms\n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//չʾͼ��
	bitmap.display_and_exit();

	cudaFree(dev_bitmap);

	return 0;
}
