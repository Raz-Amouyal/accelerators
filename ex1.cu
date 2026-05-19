#include "ex1.h"

#define MAP_SIZE (TILE_COUNT * TILE_COUNT * 256)
#define IMG_SIZE (IMG_HEIGHT * IMG_WIDTH)

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
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of
 *             the tiles' maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps, int *histogram) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int stride = blockDim.x;

    uchar *in_img    = all_in    + bid * IMG_SIZE;
    uchar *out_img   = all_out   + bid * IMG_SIZE;
    uchar *img_maps  = maps      + bid * MAP_SIZE;
    int   *img_hist  = histogram + bid * MAP_SIZE;

    // initialize histogram
    for (int i = tid; i < MAP_SIZE; i += stride) {
        img_hist[i] = 0;
    }
    __syncthreads();

    // compute histogram (one bin per tile, per gray level)
    for (int i = tid; i < IMG_SIZE; i += stride) {
        int row = i / IMG_WIDTH;
        int col = i % IMG_WIDTH;
        int tile_row = row / TILE_WIDTH;
        int tile_col = col / TILE_WIDTH;
        int tile_idx = tile_row * TILE_COUNT + tile_col;
        atomicAdd(&img_hist[tile_idx * 256 + in_img[i]], 1);
    }
    __syncthreads();

    // compute CDF (in-place prefix sum over each tile's 256-bin histogram)
    for (int t = 0; t < TILE_COUNT * TILE_COUNT; t++) {
        prefix_sum(&img_hist[t * 256], 256);
    }
    __syncthreads();

    // compute maps: m[v] = floor(CDF[v] / (T*T) * 255)
    for (int i = tid; i < MAP_SIZE; i += stride) {
        img_maps[i] = (uchar)((img_hist[i] * 255) / (TILE_WIDTH * TILE_WIDTH));
    }
    __syncthreads();

    // perform interpolation
    interpolate_device(img_maps, in_img, out_img);
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    uchar *in_img_gpu;
    uchar *out_img_gpu;
    uchar *maps_gpu;
    int   *histogram_gpu;
};

/* Allocate GPU memory for a single input image and a single output image.
 *
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    CUDA_CHECK(cudaMalloc(&context->in_img_gpu,    IMG_SIZE * sizeof(uchar)));
    CUDA_CHECK(cudaMalloc(&context->out_img_gpu,   IMG_SIZE * sizeof(uchar)));
    CUDA_CHECK(cudaMalloc(&context->maps_gpu,      MAP_SIZE * sizeof(uchar)));
    CUDA_CHECK(cudaMalloc(&context->histogram_gpu, MAP_SIZE * sizeof(int)));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    int threadblock_size = 1024;
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img_in  = &images_in [i * IMG_SIZE];
        uchar *img_out = &images_out[i * IMG_SIZE];

        CUDA_CHECK(cudaMemcpy(context->in_img_gpu, img_in,
                              IMG_SIZE * sizeof(uchar), cudaMemcpyHostToDevice));

        process_image_kernel<<<1, threadblock_size>>>(
            context->in_img_gpu, context->out_img_gpu,
            context->maps_gpu,   context->histogram_gpu);

        CUDA_CHECK(cudaMemcpy(img_out, context->out_img_gpu,
                              IMG_SIZE * sizeof(uchar), cudaMemcpyDeviceToHost));
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    CUDA_CHECK(cudaFree(context->in_img_gpu));
    CUDA_CHECK(cudaFree(context->out_img_gpu));
    CUDA_CHECK(cudaFree(context->maps_gpu));
    CUDA_CHECK(cudaFree(context->histogram_gpu));
    delete context;
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    uchar *all_in_gpu;
    uchar *all_out_gpu;
    uchar *maps_gpu;
    int   *histogram_gpu;
};

/* Allocate GPU memory for all the input images, output images, and maps.
 *
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    CUDA_CHECK(cudaMalloc(&context->all_in_gpu,    (size_t)N_IMAGES * IMG_SIZE * sizeof(uchar)));
    CUDA_CHECK(cudaMalloc(&context->all_out_gpu,   (size_t)N_IMAGES * IMG_SIZE * sizeof(uchar)));
    CUDA_CHECK(cudaMalloc(&context->maps_gpu,      (size_t)N_IMAGES * MAP_SIZE * sizeof(uchar)));
    CUDA_CHECK(cudaMalloc(&context->histogram_gpu, (size_t)N_IMAGES * MAP_SIZE * sizeof(int)));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    int threadblock_size = 1024;

    CUDA_CHECK(cudaMemcpy(context->all_in_gpu, images_in,
                          (size_t)N_IMAGES * IMG_SIZE * sizeof(uchar),
                          cudaMemcpyHostToDevice));

    process_image_kernel<<<N_IMAGES, threadblock_size>>>(
        context->all_in_gpu, context->all_out_gpu,
        context->maps_gpu,   context->histogram_gpu);

    CUDA_CHECK(cudaMemcpy(images_out, context->all_out_gpu,
                          (size_t)N_IMAGES * IMG_SIZE * sizeof(uchar),
                          cudaMemcpyDeviceToHost));
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    CUDA_CHECK(cudaFree(context->all_in_gpu));
    CUDA_CHECK(cudaFree(context->all_out_gpu));
    CUDA_CHECK(cudaFree(context->maps_gpu));
    CUDA_CHECK(cudaFree(context->histogram_gpu));
    delete context;
}
