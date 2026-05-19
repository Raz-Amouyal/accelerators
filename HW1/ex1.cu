#include "ex1.h"

#define MAP_SIZE    (TILE_COUNT * TILE_COUNT * 256)
#define IMG_SIZE    (IMG_HEIGHT * IMG_WIDTH)
#define TILE_PIXELS (TILE_WIDTH * TILE_WIDTH)
#define HIST_SIZE   256

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int inc;
    for (int stride = 1; stride < arr_size; stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            inc = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            arr[tid] += inc;
        }
        __syncthreads();
    }
}

/**
 * Perform interpolation on a single image
 */
__device__
void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) {
    __shared__ int s_hist[HIST_SIZE];

    int tid    = threadIdx.x;
    int bid    = blockIdx.x;
    int stride = blockDim.x;

    uchar *in_img   = all_in  + bid * IMG_SIZE;
    uchar *out_img  = all_out + bid * IMG_SIZE;
    uchar *img_maps = maps    + bid * MAP_SIZE;

    // process each tile in turn, keeping only a 1 KB histogram in shared memory
    for (int t = 0; t < TILE_COUNT * TILE_COUNT; t++) {
        int t_row  = t / TILE_COUNT;
        int t_col  = t % TILE_COUNT;
        int origin = t_row * TILE_WIDTH * IMG_WIDTH + t_col * TILE_WIDTH;

        // zero the shared histogram
        if (tid < HIST_SIZE) s_hist[tid] = 0;
        __syncthreads();

        // build histogram for this tile (shared-memory atomics — much faster than global)
        for (int i = tid; i < TILE_PIXELS; i += stride) {
            int row = i >> TILE_WIDTH_LOG2;          // i / TILE_WIDTH
            int col = i & (TILE_WIDTH - 1);          // i % TILE_WIDTH (T is power of 2)
            atomicAdd(&s_hist[in_img[origin + row * IMG_WIDTH + col]], 1);
        }
        __syncthreads();

        // CDF: in-place prefix sum
        prefix_sum(s_hist, HIST_SIZE);

        // write the tile's map: m[v] = floor(CDF[v] / (T*T) * 255)
        if (tid < HIST_SIZE) {
            img_maps[t * HIST_SIZE + tid] =
                (uchar)((s_hist[tid] * 255) / TILE_PIXELS);
        }
        __syncthreads();
    }

    interpolate_device(img_maps, in_img, out_img);
}

/* Task serial */
struct task_serial_context {
    uchar *in_img_gpu;
    uchar *out_img_gpu;
    uchar *maps_gpu;
};

struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;
    CUDA_CHECK(cudaMalloc(&context->in_img_gpu,  IMG_SIZE * sizeof(uchar)));
    CUDA_CHECK(cudaMalloc(&context->out_img_gpu, IMG_SIZE * sizeof(uchar)));
    CUDA_CHECK(cudaMalloc(&context->maps_gpu,    MAP_SIZE * sizeof(uchar)));
    return context;
}

void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    int threadblock_size = 1024;
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img_in  = &images_in [i * IMG_SIZE];
        uchar *img_out = &images_out[i * IMG_SIZE];

        CUDA_CHECK(cudaMemcpy(context->in_img_gpu, img_in,
                              IMG_SIZE * sizeof(uchar), cudaMemcpyHostToDevice));

        process_image_kernel<<<1, threadblock_size>>>(
            context->in_img_gpu, context->out_img_gpu, context->maps_gpu);

        CUDA_CHECK(cudaMemcpy(img_out, context->out_img_gpu,
                              IMG_SIZE * sizeof(uchar), cudaMemcpyDeviceToHost));
    }
}

void task_serial_free(struct task_serial_context *context)
{
    CUDA_CHECK(cudaFree(context->in_img_gpu));
    CUDA_CHECK(cudaFree(context->out_img_gpu));
    CUDA_CHECK(cudaFree(context->maps_gpu));
    delete context;
}

/* Bulk GPU */
struct gpu_bulk_context {
    uchar *all_in_gpu;
    uchar *all_out_gpu;
    uchar *maps_gpu;
};

struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;
    CUDA_CHECK(cudaMalloc(&context->all_in_gpu,  (size_t)N_IMAGES * IMG_SIZE * sizeof(uchar)));
    CUDA_CHECK(cudaMalloc(&context->all_out_gpu, (size_t)N_IMAGES * IMG_SIZE * sizeof(uchar)));
    CUDA_CHECK(cudaMalloc(&context->maps_gpu,    (size_t)N_IMAGES * MAP_SIZE * sizeof(uchar)));
    return context;
}

void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    int threadblock_size = 1024;

    CUDA_CHECK(cudaMemcpy(context->all_in_gpu, images_in,
                          (size_t)N_IMAGES * IMG_SIZE * sizeof(uchar),
                          cudaMemcpyHostToDevice));

    process_image_kernel<<<N_IMAGES, threadblock_size>>>(
        context->all_in_gpu, context->all_out_gpu, context->maps_gpu);

    CUDA_CHECK(cudaMemcpy(images_out, context->all_out_gpu,
                          (size_t)N_IMAGES * IMG_SIZE * sizeof(uchar),
                          cudaMemcpyDeviceToHost));
}

void gpu_bulk_free(struct gpu_bulk_context *context)
{
    CUDA_CHECK(cudaFree(context->all_in_gpu));
    CUDA_CHECK(cudaFree(context->all_out_gpu));
    CUDA_CHECK(cudaFree(context->maps_gpu));
    delete context;
}
