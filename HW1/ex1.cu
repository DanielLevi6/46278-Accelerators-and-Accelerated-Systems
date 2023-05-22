#include "ex1.h"

__device__ void prefixSum(int arr[], int len, int tid, int threads)
{
    //TODO
    int increment;

    for(int stride = 1; stride < threads; stride *= 2)
    {
        if(tid >= stride && tid < len)
        {
            increment = arr[tid - stride];
        }

        __syncthreads();

        if(tid >= stride && tid < len)
        {
            arr[tid] += increment;
        }

        __syncthreads();
    }
}

__device__ void argmin(int arr[], int len, int tid, int threads)
{
    assert(threads == len / 2);
    int halfLen = len / 2;
    bool firstIteration = true;
    int prevHalfLength = 0;
    while (halfLen > 0)
    {
        if (tid < halfLen)
        {
            if (arr[tid] == arr[tid + halfLen])
            { // a corner case
                int lhsIdx = tid;
                int rhdIdx = tid + halfLen;
                int lhsOriginalIdx = firstIteration ? lhsIdx : arr[prevHalfLength + lhsIdx];
                int rhsOriginalIdx = firstIteration ? rhdIdx : arr[prevHalfLength + rhdIdx];
                arr[tid + halfLen] = lhsOriginalIdx < rhsOriginalIdx ? lhsOriginalIdx : rhsOriginalIdx;
            }
            else
            { // the common case
                bool isLhsSmaller = (arr[tid] < arr[tid + halfLen]);
                int idxOfSmaller = isLhsSmaller * tid + (!isLhsSmaller) * (tid + halfLen);
                int smallerValue = arr[idxOfSmaller];
                int origIdxOfSmaller = firstIteration * idxOfSmaller + (!firstIteration) * arr[prevHalfLength + idxOfSmaller];
                arr[tid] = smallerValue;
                arr[tid + halfLen] = origIdxOfSmaller;
            }
        }
        __syncthreads();
        firstIteration = false;
        prevHalfLength = halfLen;
        halfLen /= 2;
    }
}

__device__ void colorHist(uchar img[][CHANNELS], int histograms[][LEVELS])
{
    //TODO
    int tid = threadIdx.x;
    int threads = blockDim.x;
    
    // Init
    if(tid < LEVELS)
    {
        histograms[0][tid] = 0;
        histograms[1][tid] = 0;
        histograms[2][tid] = 0;
    }

    __syncthreads();

    for(int i = tid; i < SIZE * SIZE; i += threads)
    {
        // Red channel
        atomicAdd(histograms[0] + img[i][0], 1);

        // Green channel
        atomicAdd(histograms[1] + img[i][1], 1);

        // Blue channel
        atomicAdd(histograms[2] + img[i][2], 1);
    }

    __syncthreads();
}

__device__ void performMapping(uchar maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS])
{
    //TODO
    int tid = threadIdx.x;
    int threads = blockDim.x;
    
    for(int i = tid; i < SIZE * SIZE; i += threads)
    {
        resultImg[i][0] = maps[0][targetImg[i][0]];
        resultImg[i][1] = maps[1][targetImg[i][1]];
        resultImg[i][2] = maps[2][targetImg[i][2]];
    }

    __syncthreads();
}

__device__ void calculateMap(uchar maps[LEVELS], int targetHist[LEVELS], int refrenceHist[LEVELS])
{
    __shared__ int diff[LEVELS];

    int tid = threadIdx.x;
    int threads = blockDim.x;

    for(int i_tar = 0; i_tar < LEVELS; i_tar++){
        for(int i_ref = tid; i_ref < LEVELS; i_ref += threads){
            diff[tid] = abs(refrenceHist[tid] - targetHist[i_tar]);
        }

        __syncthreads();

        argmin(diff, LEVELS, tid, LEVELS/2);
        
        __syncthreads();

        if(tid == 0)
        {
            maps[i_tar] = diff[1];
        }
        
        __syncthreads();
    }

}

__global__
void process_image_kernel(uchar *targets, uchar *refrences, uchar *results) {
    // TODO
    __shared__ int targetHist[CHANNELS][LEVELS];
    __shared__ int refrenceHist[CHANNELS][LEVELS];
    __shared__ uchar maps[CHANNELS][LEVELS];

    int tid = threadIdx.x;
    int threads = blockDim.x;
    int bid = blockIdx.x;

    // Step 1 - For each image (target and reference), create a histogram
    colorHist(reinterpret_cast<uchar(*) [CHANNELS]>(&targets[bid * SIZE * SIZE * CHANNELS]), targetHist);
    colorHist(reinterpret_cast<uchar(*) [CHANNELS]>(&refrences[bid * SIZE * SIZE * CHANNELS]), refrenceHist);

    __syncthreads();
    
    // Step 2 - Calculate the prefix sum of the histogram
    for (int i = 0; i < CHANNELS; i++){
        prefixSum(targetHist[i], LEVELS, tid, threads);
        prefixSum(refrenceHist[i], LEVELS, tid, threads);
    }

    __syncthreads();

    // Step 3 - Calculate a map 𝑚 from old to new gray levels
    for (int i = 0; i < CHANNELS; i++){
        calculateMap(maps[i], targetHist[i], refrenceHist[i]);
    }

    __syncthreads();

    // Step 4 - Perform the mapping process
    performMapping(maps, reinterpret_cast<uchar(*)[CHANNELS]>(&targets[bid * SIZE * SIZE * CHANNELS]), reinterpret_cast<uchar(*)[CHANNELS]>(&results[bid * SIZE * SIZE * CHANNELS]));
    
    __syncthreads();
}


/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    // TODO define task serial memory buffers
    uchar* targets;
    uchar* references;
    uchar* results;
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context* task_serial_init()
{
    auto context = new task_serial_context;

    //TODO: allocate GPU memory for a single input image and a single output image
    cudaMalloc(&context->targets, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar));
    cudaMalloc(&context->references, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar));
    cudaMalloc(&context->results, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_target, uchar *images_refrence, uchar *images_result)
{
    //TODO: in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
    for(int i = 0; i < N_IMAGES; i++)
    {
        cudaMemcpy(&context->targets[i * SIZE * SIZE * CHANNELS], &images_target[i * SIZE * SIZE * CHANNELS], SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyHostToDevice);
        cudaMemcpy(&context->references[i * SIZE * SIZE * CHANNELS], &images_refrence[i * SIZE * SIZE * CHANNELS], SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyHostToDevice);
        
        cudaDeviceSynchronize();

        process_image_kernel<<<1, 1024>>>(&(context->targets[i * SIZE * SIZE * CHANNELS]), &(context->references[i * SIZE * SIZE * CHANNELS]), &(context->results[i * SIZE * SIZE * CHANNELS]));
    
        cudaDeviceSynchronize();

        cudaMemcpy(&images_result[i * SIZE * SIZE * CHANNELS], &context->results[i * SIZE * SIZE * CHANNELS], SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyDeviceToHost);
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init
    cudaFree(context->targets);
    cudaFree(context->references);
    cudaFree(context->results);

    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
    uchar* targets;
    uchar* references;
    uchar* results;
};

/* Allocate GPU memory for all the input and output images.
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all input images and all output images
    cudaMalloc(&context->targets, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar));
    cudaMalloc(&context->references, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar));
    cudaMalloc(&context->results, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_target, uchar *images_refrence, uchar *images_result)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out
    cudaMemcpy(context->targets, images_target, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(context->references, images_refrence, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyHostToDevice);
    
    //cudaDeviceSynchronize();

    process_image_kernel<<<N_IMAGES, 1024>>>(context->targets, context->references, context->results);

    cudaDeviceSynchronize();

    cudaMemcpy(images_result, context->results, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyDeviceToHost);
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init
    cudaFree(context->targets);
    cudaFree(context->references);
    cudaFree(context->results);

    free(context);
}


/********************************************************
**  the following waappers are needed for unit testing.
********************************************************/

__global__ void argminWrapper(int arr[], int size){
    argmin(arr, size, threadIdx.x, blockDim.x);
}

__global__ void colorHistWrapper(uchar img[][CHANNELS], int histograms[][LEVELS]){
    __shared__ int histogramsSahred[CHANNELS][LEVELS];

    int tid = threadIdx.x;;
    int threads = blockDim.x;

    colorHist(img, histogramsSahred);

    __syncthreads();

    for(int i = tid; i < CHANNELS * LEVELS; i+=threads){
        ((int*)histograms)[i] = ((int*)histogramsSahred)[i];
    }
    
}

__global__ void prefixSumWrapper(int arr[], int size){
    __shared__ int arrSahred[LEVELS];

    int tid = threadIdx.x;
    int threads = blockDim.x;

    for(int i=tid; i<size; i+=threads){
        arrSahred[i] = arr[i];
    }

    __syncthreads();

    prefixSum(arrSahred, size, threadIdx.x, blockDim.x);

    for(int i=tid; i<size; i+=threads){
        arr[i] = arrSahred[i];
    }

    __syncthreads();
}

__global__ void performMappingWrapper(uchar maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS]){
    performMapping(maps, targetImg, resultImg);
}
