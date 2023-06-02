#include "ex2.h"
#include <cuda/atomic>

#define NO_ID (-1)

typedef struct{
    cudaStream_t stream;

    bool available;

    uchar *target;
    uchar *reference;
    uchar *result;
    int job_id;
} stream_struct;

/***************************** Image processing funcs *****************************/
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
    //TODO
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

__device__
void process_image(uchar *target, uchar *reference, uchar *result) {
    // TODO complete according to hw1
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

class streams_server : public image_processing_server
{
private:
    // TODO define stream server context (memory buffers, streams, etc...)
    stream_struct stream_contexts[STREAM_COUNT];

public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
        for(int i=0; i<STREAM_COUNT; i++)
        {
            // Stream init
            CUDA_CHECK(cudaStreamCreate(&stream_contexts[i].stream));

            // Buffer init
            CUDA_CHECK(cudaMalloc(&stream_contexts[i].target, SIZE * SIZE * CHANNELS * sizeof(uchar)));
            CUDA_CHECK(cudaMalloc(&stream_contexts[i].reference, SIZE * SIZE * CHANNELS * sizeof(uchar)));
            CUDA_CHECK(cudaMalloc(&stream_contexts[i].result, SIZE * SIZE * CHANNELS * sizeof(uchar)));
            stream_contexts[i].job_id = NO_ID;

            // Availability init
            stream_contexts[i].available = true;
        }
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
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
        // TODO place memory transfers and kernel invocation in streams if possible.
        for(int i=0; i<STREAM_COUNT; i++)
        {
            if(stream_contexts[i].available)
            {
                stream_contexts[i].job_id = job_id;
                stream_contexts[i].available = false;
                CUDA_CHECK(cudaMemcpyAsync(stream_contexts[i].target, target, SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyHostToDevice, stream_contexts[i].stream));
                CUDA_CHECK(cudaMemcpyAsync(stream_contexts[i].reference, reference, SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyHostToDevice, stream_contexts[i].stream));
                process_image_kernel<<<1, 1024, 0, stream_contexts[i].stream>>>(stream_contexts[i].target, stream_contexts[i].reference, stream_contexts[i].result);
                CUDA_CHECK(cudaMemcpyAsync(result, stream_contexts[i].result, SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyDeviceToHost, stream_contexts[i].stream));
                return true;
            }
        }
        return false;
    }

    bool dequeue(int *job_id) override
    {
        // TODO query (don't block) streams for any completed requests.
        for(int i=0; i<STREAM_COUNT; i++)
        {
            if(stream_contexts[i].job_id != NO_ID)
            {
                cudaError_t status = cudaStreamQuery(stream_contexts[i].stream); // TODO query diffrent stream each iteration
                switch (status) {
                case cudaSuccess:
                    // TODO return the img_id of the request that was completed.
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

// TODO implement a SPSC queue
class SpscQueue
{

};

// template <typename T, uint8_t size> class ring_buffer {
// private:
//     static const size_t N = 1 << size;
//     T _mailbox[N];
//     cuda::atomic<size_t> _head = 0, _tail = 0;
// public:
//     void push(const T &data) {
//         int tail = _tail.load(memory_order_relaxed);
//         while (tail - _head.load(memory_order_acquire) == N);
//         _mailbox[_tail % N] = data;
//         _tail.store(tail + 1, memory_order_release);
//     }
//     T pop() {
//         int head = _head.load(memory_order_relaxed);
//         while (_tail.load(memory_order_acquire) == _head);
        
//         T item = _mailbox[_head % N];
//         _head.store(head + 1, memory_order_release);
//         return item;
//     }
// };

// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
    SpscQueue cpu_to_gpu_queue;
    SpscQueue gpu_to_cpu_queue;
public:
    queue_server(int threads)
    {
        // TODO initialize host state
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
    }

    bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        // TODO push new task into queue if possible
        return false;
    }

    bool dequeue(int *job_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        return false;

        // TODO return the job_id of the request that was completed.
        //*job_id = ... 
        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
