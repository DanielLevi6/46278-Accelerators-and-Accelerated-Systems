#include "ex2.h"
#include <cuda/atomic>

#define NO_ID (-1)
#define STOP (-2)
#define QUEUE_SLOTS 4
#define REGISTERS_PER_THREAD 32
/*
Each block calculates an image

In calculateMap-    __shared__ int diff[LEVELS]
In process_image-   __shared__ int targetHist[CHANNELS][LEVELS];
                    __shared__ int refrenceHist[CHANNELS][LEVELS];
                    __shared__ uchar maps[CHANNELS][LEVELS];
*/
#define SHMEM_PER_BLOCK ((LEVELS * sizeof(int)) + 2 * (LEVELS * CHANNELS * sizeof(int)) + (LEVELS * CHANNELS * sizeof(uchar)) + sizeof(QueueRequest))

/**********************************************************************************/
/************************ Message structs for the servers *************************/
/**********************************************************************************/
typedef struct{
    cudaStream_t stream;

    bool available;

    uchar *target;
    uchar *reference;
    uchar *result;
    int job_id;
} StreamRequest;

typedef struct{
    uchar *target;
    uchar *reference;
    uchar *result;
    int image_id;
} QueueRequest;

/********************* Message structs for the servers - end **********************/


/**********************************************************************************/
/***************************** Image processing funcs *****************************/
/**********************************************************************************/

__device__ void prefixSum(int arr[], int len, int tid, int threads)
{
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
    int tid = threadIdx.x;
    int threads = blockDim.x;
    
    // Init the histograms
    if(tid < LEVELS)
    {
        histograms[0][tid] = 0; // Red channel
        histograms[1][tid] = 0; // Green channel
        histograms[2][tid] = 0; // Blue channel
    }

    __syncthreads();

    for(int i = tid; i < SIZE * SIZE; i += threads)
    {
        // Red channel
        atomicAdd_block(histograms[0] + img[i][0], 1);

        // Green channel
        atomicAdd_block(histograms[1] + img[i][1], 1);

        // Blue channel
        atomicAdd_block(histograms[2] + img[i][2], 1);
    }

    __syncthreads();
}

__device__ void performMapping(uchar maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS])
{
    int tid = threadIdx.x;
    int threads = blockDim.x;
    
    for(int i = tid; i < SIZE * SIZE; i += threads)
    {
        resultImg[i][0] = maps[0][targetImg[i][0]]; // Red channel
        resultImg[i][1] = maps[1][targetImg[i][1]]; // Green channel
        resultImg[i][2] = maps[2][targetImg[i][2]]; // Blue channel
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
            diff[i_ref] = abs(refrenceHist[i_ref] - targetHist[i_tar]);
        }

        __syncthreads();

        argmin(diff, LEVELS, tid, LEVELS/2);
        
        __syncthreads();

        if(tid == 0) // For preventing bank conflicts
        {
            maps[i_tar] = diff[1];
        }
        
        __syncthreads();
    }

}
/************************** Image processing funcs - end **************************/


/**********************************************************************************/
/****************************** Process image kernel ******************************/
/**********************************************************************************/

__device__
void process_image(uchar *target, uchar *reference, uchar *result) {
    __shared__ int targetHist[CHANNELS][LEVELS];
    __shared__ int refrenceHist[CHANNELS][LEVELS];
    __shared__ uchar maps[CHANNELS][LEVELS];

    int tid = threadIdx.x;
    int threads = blockDim.x;

    // Step 1 - For each image (target and reference), create a histogram
    colorHist(reinterpret_cast<uchar(*) [CHANNELS]>(target), targetHist);
    colorHist(reinterpret_cast<uchar(*) [CHANNELS]>(reference), refrenceHist);

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
    performMapping(maps, reinterpret_cast<uchar(*)[CHANNELS]>(target), reinterpret_cast<uchar(*)[CHANNELS]>(result));
    
    __syncthreads();
}

__global__
void process_image_kernel(uchar *target, uchar *reference, uchar *result){
    process_image(target, reference, result);
}

/*************************** Process image kernel - end ***************************/


/**********************************************************************************/
/************************** Stream server implementation **************************/
/**********************************************************************************/

class streams_server : public image_processing_server
{
private:
    StreamRequest stream_contexts[STREAM_COUNT];

public:
    streams_server()
    {
        for(int i=0; i<STREAM_COUNT; i++)
        {
            // Stream init
            CUDA_CHECK(cudaStreamCreate(&stream_contexts[i].stream));

            // Buffer init
            CUDA_CHECK(cudaMalloc(&stream_contexts[i].target, IMG_BYTES * sizeof(uchar)));
            CUDA_CHECK(cudaMalloc(&stream_contexts[i].reference, IMG_BYTES * sizeof(uchar)));
            CUDA_CHECK(cudaMalloc(&stream_contexts[i].result, IMG_BYTES * sizeof(uchar)));
            stream_contexts[i].job_id = NO_ID;

            // Availability init
            stream_contexts[i].available = true;
        }
    }

    ~streams_server() override
    {
        for(int i=0; i<STREAM_COUNT; i++)
        {
            // Stream free
            CUDA_CHECK(cudaStreamDestroy(stream_contexts[i].stream));

            // Buffer free
            CUDA_CHECK(cudaFree(stream_contexts[i].target));
            CUDA_CHECK(cudaFree(stream_contexts[i].reference));
            CUDA_CHECK(cudaFree(stream_contexts[i].result));
        }
    }

    bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        for(int i=0; i<STREAM_COUNT; i++)
        {
            if(stream_contexts[i].available)
            {
                stream_contexts[i].job_id = job_id;
                stream_contexts[i].available = false;
                CUDA_CHECK(cudaMemcpyAsync(stream_contexts[i].target, target, IMG_BYTES * sizeof(uchar), cudaMemcpyHostToDevice, stream_contexts[i].stream));
                CUDA_CHECK(cudaMemcpyAsync(stream_contexts[i].reference, reference, IMG_BYTES * sizeof(uchar), cudaMemcpyHostToDevice, stream_contexts[i].stream));
                process_image_kernel<<<1, 1024, 0, stream_contexts[i].stream>>>(stream_contexts[i].target, stream_contexts[i].reference, stream_contexts[i].result);
                CUDA_CHECK(cudaMemcpyAsync(result, stream_contexts[i].result, IMG_BYTES * sizeof(uchar), cudaMemcpyDeviceToHost, stream_contexts[i].stream));
                return true;
            }
        }
        return false;
    }

    bool dequeue(int *job_id) override
    {
        for(int i=0; i<STREAM_COUNT; i++)
        {
            if(stream_contexts[i].job_id != NO_ID)
            {
                cudaError_t status = cudaStreamQuery(stream_contexts[i].stream);
                switch (status) {
                case cudaSuccess:
                    *job_id = stream_contexts[i].job_id;
                    stream_contexts[i].available = true;
                    stream_contexts[i].job_id = NO_ID;
                    return true;
                case cudaErrorNotReady:
                    continue;
                default:
                    CUDA_CHECK(status);
                    return false;
                }
            }
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

/*********************** Stream server implementation - end ***********************/


/**********************************************************************************/
/*************************** SPSC Queue implementation ****************************/
/**********************************************************************************/

class RingBuffer 
{
private:
    size_t N;
    QueueRequest* _requests;
    cuda::atomic<size_t> _head, _tail;

public:
    __host__ RingBuffer() : _head(0), _tail(0), N(QUEUE_SLOTS) {
        CUDA_CHECK(cudaMallocHost(&_requests, N * sizeof(QueueRequest)));
    }

    __host__ ~RingBuffer() {
        CUDA_CHECK(cudaFreeHost(_requests));
    }

    __device__ __host__ bool push(const QueueRequest &new_request) {
        int tail = _tail.load(cuda::memory_order_relaxed);
        if(tail - _head.load(cuda::memory_order_acquire) == N)
        {
            return false;
        }
        _requests[_tail % N] = new_request;
        _tail.store(tail + 1, cuda::memory_order_release);
        return true;
    }

    __device__ __host__ QueueRequest pop() {
        QueueRequest item;
        item.image_id = NO_ID;
        int head = _head.load(cuda::memory_order_relaxed);
        if(_tail.load(cuda::memory_order_acquire) != _head)
        {
            item = _requests[_head % N];
            _requests[_head % N].image_id = NO_ID;
            _head.store(head + 1, cuda::memory_order_release);
        }
        return item;
    }
};

__global__ void persistent_processing_kernel(RingBuffer* cpu_to_gpu_queues, RingBuffer* gpu_to_cpu_queues)
{
    __shared__ QueueRequest request;
    
    request.image_id = NO_ID;

    while(1)
    {
        if(threadIdx.x == 0)
        {
            request = cpu_to_gpu_queues[blockIdx.x].pop();
        }
        __syncthreads();

        if(request.image_id == STOP)
        {
            break;
        }

        if(request.image_id != NO_ID && request.image_id != STOP)
        {
            process_image(request.target, request.reference, request.result);
            __syncthreads();

            if(threadIdx.x == 0)
            {
                while(!gpu_to_cpu_queues[blockIdx.x].push(request));
            }
            __syncthreads();
        }
    }
}

int calcTBcount(int threads_per_tb)
{
    cudaDeviceProp device;
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));

    int registers_per_tb = threads_per_tb * REGISTERS_PER_THREAD;
    int shmem_per_tb = SHMEM_PER_BLOCK;

    int max_tb_per_sm_registers = device.regsPerMultiprocessor / registers_per_tb;
    int max_tb_per_sm_shmem = device.sharedMemPerMultiprocessor / shmem_per_tb;
    int max_tb_per_sm_threads = device.maxThreadsPerMultiProcessor / threads_per_tb;
    
    return min(max_tb_per_sm_registers, min(max_tb_per_sm_shmem, max_tb_per_sm_threads)) * device.multiProcessorCount;
}

class queue_server : public image_processing_server
{
private:
    RingBuffer* cpu_to_gpu_queues;
    char* cpu_to_gpu_queues_buf;
    RingBuffer* gpu_to_cpu_queues;
    char* gpu_to_cpu_queues_buf;

    int curr_block;
    int blocks;
public:
    queue_server(int threads)
    {
        curr_block = 0;
        blocks = calcTBcount(threads);

        // Initialize host state
        CUDA_CHECK(cudaMallocHost(&cpu_to_gpu_queues_buf, blocks * sizeof(RingBuffer)));
        cpu_to_gpu_queues = reinterpret_cast<RingBuffer*>(cpu_to_gpu_queues_buf);
        CUDA_CHECK(cudaMallocHost(&gpu_to_cpu_queues_buf, blocks * sizeof(RingBuffer)));
        gpu_to_cpu_queues = reinterpret_cast<RingBuffer*>(gpu_to_cpu_queues_buf);
        
        for(int i=0; i<blocks; i++)
        {
            new(&cpu_to_gpu_queues[i]) RingBuffer();
            new(&gpu_to_cpu_queues[i]) RingBuffer();
        }

        persistent_processing_kernel<<<blocks, threads>>>(cpu_to_gpu_queues, gpu_to_cpu_queues);
    }

    ~queue_server() override
    {
        // Sends kill signal to the threads
        for(int i=0; i<blocks; i++)
        {
            this->enqueue(STOP, nullptr, nullptr, nullptr);
        }

        // Free resources allocated in constructor
        for(int i=0; i<blocks; i++)
        {
            cpu_to_gpu_queues[i].~RingBuffer();
            gpu_to_cpu_queues[i].~RingBuffer();
        }
        CUDA_CHECK(cudaFreeHost(cpu_to_gpu_queues_buf));
        CUDA_CHECK(cudaFreeHost(gpu_to_cpu_queues_buf));
    }

    bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        QueueRequest new_request;
        new_request.target = target;
        new_request.reference = reference;
        new_request.result = result;
        new_request.image_id = job_id;

        bool res = cpu_to_gpu_queues[curr_block].push(new_request);
        // For splitting the requests to all the queues equally
        curr_block = (curr_block + 1) % blocks;

        return res;
    }

    bool dequeue(int *job_id) override
    {
        QueueRequest request;
        request.image_id = NO_ID;

        for(int i=0; i<blocks; i++)
        {
            request = gpu_to_cpu_queues[i].pop();
            if(request.image_id == NO_ID){
                continue;
            }
            // Return the job_id of the request that was completed.
            *job_id = request.image_id;
            return true;
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}

/************************ SPSC Queue implementation - end *************************/
