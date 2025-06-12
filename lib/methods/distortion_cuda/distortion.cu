// Complex numbers for device
#include <cuComplex.h>


// --KERNEL--: blurs image
__global__ void kernel_blur_image(cuDoubleComplex *image_d, cuDoubleComplex *otf_batch_d, size_t h, size_t w)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute indexes
        size_t pixel_index = i*w + j; // pixel index
        size_t rgb_pixel_index = pixel_index*3; // each RGB pixel holds 3 values
        size_t batch_index = pixel_index*9; // in batch each "pixel" is a 3x3 matrix

        // Temporary storage
        cuDoubleComplex new_rgb[3] = {make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,0)};
        // Blur formula
        for (int channel_i = 0; channel_i < 3; channel_i++)
        {
            for (int channel_j = 0; channel_j < 3; channel_j++)
            {
                // Each image channel spectrum is a sum of image channel spectrum and batch OTF multiplications
                new_rgb[channel_i] = cuCadd(new_rgb[channel_i],
                    cuCmul(image_d[rgb_pixel_index + channel_j], otf_batch_d[batch_index + channel_i*3 + channel_j]));
            }
        }
        // Copy result
        for (int channel = 0; channel < 3; channel++)
        {
            image_d[rgb_pixel_index + channel] = new_rgb[channel];
        }
    }
}