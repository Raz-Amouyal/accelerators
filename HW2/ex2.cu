#include "ex2.h"
#include <cuda/atomic>
// ===== MY EDIT BEGIN: extra headers used by my additions =====
#include <new>          // placement new for queues in pinned memory
#include <algorithm>    // std::min / std::max
// ===== MY EDIT END =====

#define HIST_SIZE   256

__device__ void prefix_sum(int arr[], int arr_size) {
    // TODO complete according to hw1
    // ===== MY EDIT BEGIN: prefix_sum from hw1, generalised for nthreads < arr_size =====
    // Hillis-Steele scan in shared memory. Works for any blockDim.x >= 1, but the
    // expected use is nthreads >= 256 (so arr_size=256 means each thread handles 1
    // element). For nthreads < 256 (e.g. 128) each thread handles a few elements;
    // we keep them in a small local buffer to avoid clobbering the array.
    int tid       = threadIdx.x;
    int nthreads  = blockDim.x;
    int local_buf[8];   // ceil(256/32) = 8 — enough for nthreads >= 32

    for (int stride = 1; stride < arr_size; stride *= 2) {
        int cnt = 0;
        for (int i = tid; i < arr_size; i += nthreads) {
            local_buf[cnt++] = (i >= stride) ? arr[i - stride] : 0;
        }
        __syncthreads();
        cnt = 0;
        for (int i = tid; i < arr_size; i += nthreads) {
            if (i >= stride)
                arr[i] += local_buf[cnt];
            cnt++;
        }
        __syncthreads();
    }
    // ===== MY EDIT END =====
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
 void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__device__
void process_image(uchar *in_img, uchar *out_img, uchar* maps) {
    // TODO complete according to hw1
    // ===== MY EDIT BEGIN: process_image from hw1 (one threadblock per image) =====
    // For each of the TILE_COUNT * TILE_COUNT tiles:
    //   1) build a 256-bin histogram in shared memory using atomicAdd
    //   2) turn the histogram into a CDF with prefix_sum()
    //   3) write the tile's map: m[v] = (CDF[v] * 255) / (TILE_WIDTH^2)
    // Finally, call the library-provided interpolate_device() to produce the output.
    __shared__ int s_hist[HIST_SIZE];

    int tid          = threadIdx.x;
    int nthreads     = blockDim.x;
    const int TPIX   = TILE_WIDTH * TILE_WIDTH;   // pixels per tile

    for (int ty = 0; ty < TILE_COUNT; ty++) {
        for (int tx = 0; tx < TILE_COUNT; tx++) {
            // zero histogram
            for (int i = tid; i < HIST_SIZE; i += nthreads)
                s_hist[i] = 0;
            __syncthreads();

            // build histogram for this tile
            for (int p = tid; p < TPIX; p += nthreads) {
                int y = ty * TILE_WIDTH + p / TILE_WIDTH;
                int x = tx * TILE_WIDTH + p % TILE_WIDTH;
                atomicAdd(&s_hist[in_img[y * IMG_WIDTH + x]], 1);
            }
            __syncthreads();

            // CDF via Hillis-Steele scan
            prefix_sum(s_hist, HIST_SIZE);

            // write tile map
            uchar *map = maps + 256 * (ty * TILE_COUNT + tx);
            for (int i = tid; i < 256; i += nthreads)
                map[i] = (uchar)((s_hist[i] * 255) / TPIX);
            __syncthreads();
        }
    }

    interpolate_device(maps, in_img, out_img);
    // ===== MY EDIT END =====
}

__global__
void process_image_kernel(uchar *in_img, uchar *out_img, uchar* maps){
    process_image(in_img, out_img, maps);
}

class streams_server : public image_processing_server
{
private:
    // TODO define stream server context (memory buffers, streams, etc...)
    // ===== MY EDIT BEGIN: per-stream device buffers & bookkeeping =====
    cudaStream_t streams[STREAM_COUNT];
    uchar       *d_in   [STREAM_COUNT];   // per-stream input  buffer  (device)
    uchar       *d_out  [STREAM_COUNT];   // per-stream output buffer  (device)
    uchar       *d_maps [STREAM_COUNT];   // per-stream maps   buffer  (device)
    int          stream_img_id[STREAM_COUNT];   // -1 == free
    int          next_dequeue_idx;              // round-robin hint for dequeue()
    // ===== MY EDIT END =====

public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
        // ===== MY EDIT BEGIN: create 64 streams + per-stream device buffers =====
        for (int i = 0; i < STREAM_COUNT; ++i) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaMalloc(&d_in  [i], IMG_WIDTH * IMG_HEIGHT));
            CUDA_CHECK(cudaMalloc(&d_out [i], IMG_WIDTH * IMG_HEIGHT));
            CUDA_CHECK(cudaMalloc(&d_maps[i], TILE_COUNT * TILE_COUNT * 256));
            stream_img_id[i] = -1;
        }
        next_dequeue_idx = 0;
        // ===== MY EDIT END =====
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
        // ===== MY EDIT BEGIN: drain & release all streams =====
        for (int i = 0; i < STREAM_COUNT; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
            CUDA_CHECK(cudaFree(d_in  [i]));
            CUDA_CHECK(cudaFree(d_out [i]));
            CUDA_CHECK(cudaFree(d_maps[i]));
        }
        // ===== MY EDIT END =====
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO place memory transfers and kernel invocation in streams if possible.
        // ORIG: return false;
        // ===== MY EDIT BEGIN: find a free stream and submit H2D / kernel / D2H =====
        for (int i = 0; i < STREAM_COUNT; ++i) {
            if (stream_img_id[i] == -1) {
                stream_img_id[i] = img_id;
                CUDA_CHECK(cudaMemcpyAsync(d_in[i], img_in,
                                           IMG_WIDTH * IMG_HEIGHT,
                                           cudaMemcpyHostToDevice, streams[i]));
                // Per spec: 1024 threads, single block per image
                process_image_kernel<<<1, 1024, 0, streams[i]>>>(
                    d_in[i], d_out[i], d_maps[i]);
                CUDA_CHECK(cudaMemcpyAsync(img_out, d_out[i],
                                           IMG_WIDTH * IMG_HEIGHT,
                                           cudaMemcpyDeviceToHost, streams[i]));
                return true;
            }
        }
        return false;
        // ===== MY EDIT END =====
    }

    bool dequeue(int *img_id) override
    {
        // ORIG: return false;   // (the original early-return is disabled below)

        // TODO query (don't block) streams for any completed requests.
        // ===== MY EDIT BEGIN: round-robin probe of all 64 streams =====
        for (int k = 0; k < STREAM_COUNT; ++k) {
            int i = (next_dequeue_idx + k) % STREAM_COUNT;
            if (stream_img_id[i] == -1) continue;

            cudaError_t status = cudaStreamQuery(streams[i]);
            switch (status) {
            case cudaSuccess:
                // TODO return the img_id of the request that was completed.
                *img_id = stream_img_id[i];
                stream_img_id[i] = -1;
                next_dequeue_idx = (i + 1) % STREAM_COUNT;
                return true;
            case cudaErrorNotReady:
                continue;
            default:
                CUDA_CHECK(status);
                return false;
            }
        }
        return false;
        // ===== MY EDIT END =====

        /* ORIG (kept for reference, replaced by the loop above):
        //for ()
        //{
            cudaError_t status = cudaStreamQuery(0); // TODO query diffrent stream each iteration
            switch (status) {
            case cudaSuccess:
                // TODO return the img_id of the request that was completed.
                //*img_id = ...
                return true;
            case cudaErrorNotReady:
                return false;
            default:
                CUDA_CHECK(status);
                return false;
            }
        //}
        */
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

// TODO implement a lock
// ===== MY EDIT BEGIN: TTAS spin-lock (cuda::atomic) =====
// The lock must reside in GPU memory because RMW (exchange / CAS) on PCIe-mapped
// host memory is not atomic across the bus. We zero-initialise via cudaMemset
// before the kernel launches.
class ttas_lock {
public:
    cuda::atomic<int> state;   // 0 = free, 1 = held — initialised by cudaMemset
    __device__ void lock() {
        while (true) {
            // "test" — cheap read until the lock looks free
            while (state.load(cuda::memory_order_relaxed) != 0) { /* spin */ }
            // "test-and-set" — atomic exchange acquires
            if (state.exchange(1, cuda::memory_order_acquire) == 0)
                return;
        }
    }
    __device__ void unlock() {
        state.store(0, cuda::memory_order_release);
    }
};
// ===== MY EDIT END =====

// TODO implement a MPMC queue
// ===== MY EDIT BEGIN: bounded ring buffer in pinned host memory =====
// Slot types for the two queues:
struct request_slot  { int img_id; uchar *in_img; uchar *out_img; };
struct response_slot { int img_id; };

// Generic ring buffer header. Slots are allocated immediately after the
// metadata in a single pinned-host-memory block. head/tail are cuda::atomic
// so they're visible across PCIe with release-acquire ordering.
template<typename T>
struct cpu_gpu_ring {
    int               capacity;   // power of two
    int               mask;       // capacity - 1
    cuda::atomic<int> head;       // next slot to pop
    cuda::atomic<int> tail;       // next slot to push
    T                *slots;      // points to slots[capacity]
};
// ===== MY EDIT END =====

// TODO implement the persistent kernel
// ===== MY EDIT BEGIN: persistent kernel — one block = one consumer-worker =====
__global__
void persistent_kernel(cpu_gpu_ring<request_slot>  *req_q,
                       cpu_gpu_ring<response_slot> *resp_q,
                       ttas_lock                   *req_pop_lock,
                       ttas_lock                   *resp_push_lock,
                       cuda::atomic<int>           *stop_flag)
{
    __shared__ request_slot task;
    __shared__ int          task_state;       // 0 = none, 1 = task, 2 = stop
    __shared__ uchar        maps[TILE_COUNT * TILE_COUNT * 256];

    while (true) {
        // ---- pop a request (thread 0 only, to avoid divergent lock ops) ----
        if (threadIdx.x == 0) {
            task_state = 0;
            while (task_state == 0) {
                req_pop_lock->lock();
                int h = req_q->head.load(cuda::memory_order_relaxed);
                int t = req_q->tail.load(cuda::memory_order_acquire);
                if (h < t) {
                    task = req_q->slots[h & req_q->mask];
                    req_q->head.store(h + 1, cuda::memory_order_release);
                    task_state = 1;
                }
                req_pop_lock->unlock();
                if (task_state == 1) break;

                if (stop_flag->load(cuda::memory_order_acquire)) {
                    task_state = 2;
                    break;
                }
            }
        }
        __syncthreads();

        if (task_state == 2) return;

        // ---- process the image (all threads cooperate) ----
        process_image(task.in_img, task.out_img, maps);
        __syncthreads();

        // ---- push the response (thread 0 only) ----
        if (threadIdx.x == 0) {
            while (true) {
                resp_push_lock->lock();
                int t = resp_q->tail.load(cuda::memory_order_relaxed);
                int h = resp_q->head.load(cuda::memory_order_acquire);
                if (t - h < resp_q->capacity) {
                    resp_q->slots[t & resp_q->mask].img_id = task.img_id;
                    resp_q->tail.store(t + 1, cuda::memory_order_release);
                    resp_push_lock->unlock();
                    break;
                }
                resp_push_lock->unlock();
            }
        }
        __syncthreads();
    }
}
// ===== MY EDIT END =====

// TODO implement a function for calculating the threadblocks count
// ===== MY EDIT BEGIN: threadblock count from device properties =====
// We compute the per-SM block count as the minimum of:
//   1) threads-per-SM / threads-per-block
//   2) regs-per-SM    / (threads-per-block * 32)      [Makefile: -maxrregcount=32]
//   3) shmem-per-SM   / shmem-per-block
// then multiply by the number of SMs.
static int calc_threadblocks(int threads_per_block)
{
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // Shared memory used per persistent block:
    //   histogram (256 ints) + maps (TILE_COUNT^2 * 256 bytes)
    //   + interpolate_device's internal 1024 bytes
    //   + task + task_state housekeeping
    const size_t shmem_per_block =
            sizeof(int) * 256                       // histogram
          + TILE_COUNT * TILE_COUNT * 256           // maps
          + 1024                                    // interpolate_device internal
          + sizeof(request_slot)                    // shared task
          + sizeof(int);                            // shared task_state

    const int regs_per_thread = 32;                 // -maxrregcount=32
    const int regs_per_block  = regs_per_thread * threads_per_block;

    int blocks_by_threads = prop.maxThreadsPerMultiProcessor / threads_per_block;
    int blocks_by_regs    = prop.regsPerMultiprocessor       / regs_per_block;
    int blocks_by_shmem   = prop.sharedMemPerMultiprocessor  / shmem_per_block;

    int blocks_per_sm = std::min({blocks_by_threads, blocks_by_regs, blocks_by_shmem});
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    return blocks_per_sm * prop.multiProcessorCount;
}

// ceil(log2(x)) helper — used for "round queue size up to next power of two"
static int next_pow2(int x)
{
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}
// ===== MY EDIT END =====

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
    // ===== MY EDIT BEGIN: queue server state =====
    int                            num_blocks;
    int                            queue_capacity;

    // Pinned host allocations (cudaMallocHost) — visible to both CPU and GPU:
    void                          *req_q_buf;       // header + slots
    void                          *resp_q_buf;      // header + slots
    cuda::atomic<int>             *stop_flag;       // 1-int kill switch

    cpu_gpu_ring<request_slot>    *req_q;
    cpu_gpu_ring<response_slot>   *resp_q;

    // Device allocations — locks must live in GPU memory (PCIe RMW not atomic):
    ttas_lock                     *d_req_pop_lock;
    ttas_lock                     *d_resp_push_lock;
    // ===== MY EDIT END =====
public:
    queue_server(int threads)
    {
        // TODO initialize host state
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
        // ===== MY EDIT BEGIN: allocate queues + locks, launch persistent kernel =====
        num_blocks     = calc_threadblocks(threads);
        // queue size = 2^ceil(log2(16 * #threadblocks))
        queue_capacity = next_pow2(16 * num_blocks);

        // ---- Request queue (CPU -> GPU): pinned host memory ----
        size_t req_bytes  = sizeof(cpu_gpu_ring<request_slot>)
                          + queue_capacity * sizeof(request_slot);
        CUDA_CHECK(cudaMallocHost(&req_q_buf, req_bytes));
        req_q = new (req_q_buf) cpu_gpu_ring<request_slot>();
        req_q->capacity = queue_capacity;
        req_q->mask     = queue_capacity - 1;
        req_q->head.store(0);
        req_q->tail.store(0);
        req_q->slots    = reinterpret_cast<request_slot*>(
                              static_cast<char*>(req_q_buf) +
                              sizeof(cpu_gpu_ring<request_slot>));

        // ---- Response queue (GPU -> CPU): pinned host memory ----
        size_t resp_bytes = sizeof(cpu_gpu_ring<response_slot>)
                          + queue_capacity * sizeof(response_slot);
        CUDA_CHECK(cudaMallocHost(&resp_q_buf, resp_bytes));
        resp_q = new (resp_q_buf) cpu_gpu_ring<response_slot>();
        resp_q->capacity = queue_capacity;
        resp_q->mask     = queue_capacity - 1;
        resp_q->head.store(0);
        resp_q->tail.store(0);
        resp_q->slots    = reinterpret_cast<response_slot*>(
                              static_cast<char*>(resp_q_buf) +
                              sizeof(cpu_gpu_ring<response_slot>));

        // ---- Stop flag: pinned host memory (load-only on GPU, store on CPU) ----
        CUDA_CHECK(cudaMallocHost(&stop_flag, sizeof(cuda::atomic<int>)));
        new (stop_flag) cuda::atomic<int>(0);

        // ---- Locks: device memory (PCIe RMW is not atomic) ----
        CUDA_CHECK(cudaMalloc(&d_req_pop_lock,    sizeof(ttas_lock)));
        CUDA_CHECK(cudaMalloc(&d_resp_push_lock,  sizeof(ttas_lock)));
        CUDA_CHECK(cudaMemset(d_req_pop_lock,   0, sizeof(ttas_lock)));
        CUDA_CHECK(cudaMemset(d_resp_push_lock, 0, sizeof(ttas_lock)));

        // ---- Launch the persistent kernel ----
        persistent_kernel<<<num_blocks, threads>>>(
            req_q, resp_q,
            d_req_pop_lock, d_resp_push_lock,
            stop_flag);
        // ===== MY EDIT END =====
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
        // ===== MY EDIT BEGIN: signal kernel to exit, then free everything =====
        stop_flag->store(1, cuda::memory_order_release);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_req_pop_lock));
        CUDA_CHECK(cudaFree(d_resp_push_lock));
        CUDA_CHECK(cudaFreeHost(stop_flag));
        CUDA_CHECK(cudaFreeHost(req_q_buf));
        CUDA_CHECK(cudaFreeHost(resp_q_buf));
        // ===== MY EDIT END =====
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        // ORIG: return false;
        // ===== MY EDIT BEGIN: single-producer push to req queue =====
        // Only the main thread enqueues, so no host-side lock is needed.
        int t = req_q->tail.load(cuda::memory_order_relaxed);
        int h = req_q->head.load(cuda::memory_order_acquire);
        if (t - h >= req_q->capacity)
            return false;

        request_slot slot = { img_id, img_in, img_out };
        req_q->slots[t & req_q->mask] = slot;
        req_q->tail.store(t + 1, cuda::memory_order_release);
        return true;
        // ===== MY EDIT END =====
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        // ORIG: return false;
        // ===== MY EDIT BEGIN: single-consumer pop from resp queue =====
        int h = resp_q->head.load(cuda::memory_order_relaxed);
        int t = resp_q->tail.load(cuda::memory_order_acquire);
        if (h >= t)
            return false;

        // TODO return the img_id of the request that was completed.
        *img_id = resp_q->slots[h & resp_q->mask].img_id;
        resp_q->head.store(h + 1, cuda::memory_order_release);
        return true;
        // ===== MY EDIT END =====
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
