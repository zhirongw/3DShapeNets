// This is modification of Alex's convolution kernel extending 2d to 3d.
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define LO8(x)      ((x) & 0x000000FF)
#define MI16(x)		(((x) & 0x0000FFFF) >> 8)
#define HI24(x)     (((x) & 0x00FFFFFF) >> 16)
#define MUL24(x,y)  ((x) * (y))
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#define MIN(a, b)   ((a) < (b) ? (a) : (b))

__global__ void kSampleMultinomial(int* output, float* distribution, float* random, int k, int n){
	
	const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(id < n){
		distribution += k * id;
		random += id;
		output += k * id;

		float preSum = 0, nowSum = 0;
		for(int i = 0; i < k; i++){
			nowSum += distribution[i];
			output[i] = random[0] >= preSum && random[0] < nowSum;
			preSum = nowSum;
		}
	}
}

__global__ void kExp(float* output, float* input, unsigned int numElements){
	const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;

	for(int i = id; i < numElements; i += numThreads)
		output[i] = __expf(input[i]);
}

__global__ void kDivide(float* output, float* leftInput, float* rightInput, unsigned int numElements){
	const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;

	for(int i = id; i < numElements; i += numThreads)
		output[i] = __fdividef(leftInput[i], rightInput[i]);
}

__global__ void kConvolve_forward(float* targets, float* images, float* filters, 
                                   const int numImages, const int numFilters,
                                   const int imgSizeZ, const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                   const int moduleStride,
                                   const int numModulesZ, const int numModulesY, const int numModulesX, const int imgStride) {
    __shared__ float shFilters[4*1][4 * 4]; // pre-load 4 pixels from 4*4 filters
    __shared__ float shImages[4*1][32 * 1]; // pre-load 4 pixels from 32*2 images
    const int imgPixels = imgSizeZ * imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize * filterSize;

    const int blocksPerModule = numFilters / (4*4);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = blockIdx.y % blocksPerModule;

    const int tidx = threadIdx.y * 32 + threadIdx.x;

    const int imgLoadModPosZ = (moduleIdx / (numModulesX * numModulesY)) * moduleStride;
    const int imgLoadModPosY = ((moduleIdx / numModulesX) % numModulesY )* moduleStride;
    const int imgLoadModPosX = (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (4 * 4);
    const int shFilterLoadX = tidx % (4 * 4);
    const int myImgIdx = blockIdx.x * 32 * 1 + threadIdx.x;
    images += myImgIdx;
    filters += 4 * 4 * blockFilterIdx
             + shFilterLoadY * numFilters + shFilterLoadX;
    targets += moduleIdx * numImages
            + (blockFilterIdx * 4 * 4 + threadIdx.y) * numImages * numModulesZ * numModulesY * numModulesX
            + myImgIdx;


    float prod[4][1];
    #pragma unroll
    for(int f = 0; f < 4; f++) {
        #pragma unroll
        for(int g = 0; g < 1; g++) {
            prod[f][g] = 0;
        }
    }

    for (int p = 0; p < filterPixels; p += 4) {
        /*
         * Load 4 pixels from 4*4 filters
         */
        if (shFilterLoadY < 4) {
            #pragma unroll
            for (int p2 = 0; p2 < 4; p2 += 32/4) {
                if (p + p2 + shFilterLoadY < filterPixels) {
                    #pragma unroll
                    for (int c = 0; c < 1; c++) {
                        shFilters[shFilterLoadY + p2 + c * 4][shFilterLoadX] = filters[(c * filterPixels + p + p2) * numFilters];
                    }
                } else {
                    #pragma unroll
                    for (int c = 0; c < 1; c++) {
                        shFilters[shFilterLoadY + p2 + c * 4][shFilterLoadX] = 0;
                    }
                }
            }
        }

        /*
         * Load 4 pixels from 32*1 images
         */
        const int pixIdx = p + threadIdx.y;
        if (pixIdx < filterPixels) {
            const int x = paddingStart + imgLoadModPosX + pixIdx % filterSize;
            const int y = paddingStart + imgLoadModPosY + (pixIdx / filterSize) % filterSize;
            const int z = paddingStart + imgLoadModPosZ + pixIdx / (filterSize * filterSize);
            if (z >= 0 && z < imgSizeZ && y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
                #pragma unroll
                for (int i = 0; i < 1; i++) {
                    if (!true || myImgIdx + i * 32 < numImages) {
                        #pragma unroll
                        for (int c = 0; c < 1; c++) {
                            shImages[threadIdx.y + c * 4][threadIdx.x + i * 32] = images[imgStride * (c * imgPixels + z * imgSizeX * imgSizeY + y * imgSizeX + x) + i * 32];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < 1; c++) {
                            shImages[threadIdx.y + c * 4][threadIdx.x + i * 32] = 0;
                        }
                    }
                }
            } else { // Padding
                #pragma unroll
                for (int i = 0; i < 1; i++) {
                    #pragma unroll
                    for (int c = 0; c < 1; c++) {
                        shImages[threadIdx.y + c * 4][threadIdx.x + i * 32] = 0;
                    }
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 4*1; i++) {
            #pragma unroll
            for(int f = 0; f < 4; f++) {
                #pragma unroll
                for(int g = 0; g < 1; g++) {
                    prod[f][g] += shImages[i][g * 32 + threadIdx.x] * shFilters[i][threadIdx.y + f * 4];
                }
            }

        }
        __syncthreads();
    }
    
    #pragma unroll
    for (int g = 0; g < 1; g++) {
        if (!true || myImgIdx + g * 32 < numImages) {
            #pragma unroll
            for (int f = 0; f < 4; f++) {
                targets[g * 32 + f * 4 * numImages * numModulesZ * numModulesY * numModulesX] = prod[f][g];
            }
        }
    }
}

__global__ void kConvolve_weight(float* targets, float* images, float* hidActs,
                                   const int numImages, const int numFilters,
                                   const int numModulesZ, const int numModulesY, const int numModulesX,
                                   const int imgSizeZ, const int imgSizeY, const int imgSizeX, const int filterSize,
                                   const int paddingStart, const int moduleStride, const int imgStride,
                                   const int partialSum, const float scaleOutputs) {
    __shared__ float shImages[5 * 8 * 1][32]; // preload 32 cases of 8 * 5 pixels
    __shared__ float shHidActs[16][32 + 1]; // preload 32 cases of 16 hidActs

    const int tidx = 16 * threadIdx.y + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;

    const int filterPixels = filterSize * filterSize * filterSize;
    const int imgPixels = imgSizeZ * imgSizeY * imgSizeX;

    const int filterBlocksPerModule = numFilters / 16;
    const int outputModuleIdx = blockIdx.x / filterBlocksPerModule;
    const int moduleIdx = partialSum * outputModuleIdx;
    const int blockFilterIdx = 16 * (blockIdx.x % filterBlocksPerModule);

//    const int moduleStride = (imgSize - filterSize + 1) / numModulesX; 
    const int numModules = numModulesZ * numModulesY * numModulesX;

    const int blockPixelOffset = blockIdx.y * 8 * 5;

    images += loadX;
    hidActs += moduleIdx * numImages
            + blockFilterIdx * numImages * numModules
            + loadY * numImages * numModules
            + loadX;
    
    targets += (outputModuleIdx * numFilters) * filterPixels * 1
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + threadIdx.y * numFilters + threadIdx.x;

    float* shImgLoad = &shImages[loadY][loadX];
    float* shHidActLoad = &shHidActs[loadY][loadX];

    float prod[1][5];
    #pragma unroll
    for (int c = 0; c < 1; c++) {
        #pragma unroll
        for (int p = 0; p < 5; p++) {
            prod[c][p] = 0;
        }
    }
    
    __shared__ int pxDivs[8*5];
    if (tidx < 8 * 5) {
        pxDivs[tidx] = (((blockPixelOffset + tidx) / (filterSize * filterSize)) << 16) + (((blockPixelOffset + tidx) / filterSize) % filterSize << 8) + ((blockPixelOffset + tidx) % filterSize);
    }
    __syncthreads();
    for (int m = moduleIdx; m < moduleIdx + partialSum; m++) {
        const int imgLoadModPosZ = paddingStart + (m / (numModulesX * numModulesY)) * moduleStride;
        const int imgLoadModPosY = paddingStart + ((m / numModulesX) % numModulesY) * moduleStride;
        const int imgLoadModPosX = paddingStart + (m % numModulesX) * moduleStride;
        for (int caseIdx = 0; caseIdx < numImages; caseIdx += 32) {
            if (loadY < 8 * 5) {
                /*
                 * As long as 8 * 16 is divisible by 32 this will loop the right
                 * number of times.
                 *
                 * This will load some imgGrads from filter pixels that don't exit (it'll set those to 0),
                 * but the code does not produce any output for those pixels (see last lines).
                 */
    //            #pragma unroll
                for (int y = 0; y < 8 * 5; y += (16 * 8) / 32) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if ((8 * 5) % (16 * 8 / 32) == 0 || y + loadY < 8 * 5) {
                        const int pxIdx = loadY + y; // pixel idx in filter

                        if (pxIdx + blockPixelOffset < filterPixels && (!true || caseIdx + loadX < numImages)) {
                            const int pxZ = imgLoadModPosZ + HI24(pxDivs[pxIdx]);
                            const int pxY = imgLoadModPosY + MI16(pxDivs[pxIdx]); // pixel x,y coords in image
                            const int pxX = imgLoadModPosX + LO8(pxDivs[pxIdx]);
                            if (pxZ >= 0 && pxZ < imgSizeZ && pxY >= 0 && pxY < imgSizeY && pxX >= 0 && pxX < imgSizeX) {
                                const int pixIdx = (pxZ * imgSizeX * imgSizeY + pxY * imgSizeX + pxX) * imgStride;
                                #pragma unroll
                                for (int c = 0; c < 1; c++) {
                                    shImgLoad[(y + c * 5 * 8) * 32] = images[caseIdx + c * imgPixels * imgStride + pixIdx];
                                }
                            } else {
                                #pragma unroll
                                for (int c = 0; c < 1; c++) {
                                    shImgLoad[(y + c * 5 * 8) * 32] = 0;
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < 1; c++) {
                                shImgLoad[(y + c * 5 * 8) * 32] = 0;
                            }
                        }
                    }
                }
            }
            if (loadY < 16 && (!true || caseIdx + loadX < numImages)) {
                #pragma unroll
                for (int y = 0; y < 16; y += (16 * 8) / 32) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if (16 % (16 * 8 / 32) == 0 || y + loadY < 16) {
                        shHidActLoad[y * (32 + 1)] = hidActs[caseIdx + y * numImages * numModules];
                    }
                }
            }

            __syncthreads();
            #pragma unroll
            for (int p = 0; p < 5; p++) {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    #pragma unroll
                    for (int c = 0; c < 1; c++) {
                        prod[c][p] += shImages[threadIdx.y + p * 8 + c * 5 * 8][i] * shHidActs[threadIdx.x][i];
                    }
                }
            }
            __syncthreads();
        }
        hidActs += numImages;
    }
    
    #pragma unroll
    for (int p = 0; p < 5; p++) {
        if (blockPixelOffset + p * 8 + threadIdx.y < filterPixels) {
            #pragma unroll
            for (int c = 0; c < 1; c++) {
                targets[p * 8 * numFilters + c * filterPixels * numFilters] = scaleOutputs * prod[c][p];
            }
        }
    }
}

__global__ void kConvolve_backward(float* targets, const float* hidActs, const float* filters, 
                                   const int numModulesZ, const int numModulesY, const int numModulesX, 
                                   const int numImages, const int numFilters, const int filterSize, 
                                   const int imgSizeZ, const int imgSizeY, const int imgSizeX,
                                   const int paddingStart, const int moduleStride) {
    __shared__ float shFilters[1*16][16 + 1]; // load 16 filter one time. See below.
    __shared__ float shHidActs[16][16*2]; // each block deal with 16 * imgPerThread images.

    const int blockCaseIdx = blockIdx.x * 16 * 2;
    const int numRegionsX = DIVUP(imgSizeX, 4);
    const int numRegionsY = DIVUP(imgSizeY, 4);
    const int blockRegionIdx = blockIdx.y;
    const int blockRegionIdxX = blockRegionIdx % numRegionsX;
    const int blockRegionIdxY = (blockRegionIdx / numRegionsX) % numRegionsY;
    const int blockRegionIdxZ = blockRegionIdx / (numRegionsX * numRegionsY);
    const int blockRegionLeft = blockRegionIdxX * 4;
    const int blockRegionTop = blockRegionIdxY * 4;
    const int blockRegionFront = blockRegionIdxZ;
    const int pxYInRegion = threadIdx.y / 4, pxXInRegion = threadIdx.y % 4;
    const int pxZ = blockRegionFront;
    const int pxY = blockRegionTop + pxYInRegion;
    const int pxX = blockRegionLeft + pxXInRegion;
    const int pxIdx = pxZ * imgSizeX * imgSizeY + pxY * imgSizeX + pxX;
    const bool isPxInImg = pxZ < imgSizeZ && pxY < imgSizeY && pxX < imgSizeX;
    const int numModules = numModulesZ * numModulesY * numModulesX;
    const int filterPixels = filterSize * filterSize * filterSize;
    const int imgPixels = imgSizeZ * imgSizeX * imgSizeY;
    const int tidx = threadIdx.y * 16 + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32; // load 32 cases one time.

    hidActs += blockCaseIdx + loadY * numImages * numModules + loadX;
    filters += threadIdx.x;
    targets += pxIdx * numImages + blockCaseIdx + threadIdx.x;


    float prod[1][2];
    #pragma unroll
    for (int c = 0; c < 1; c++) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            prod[c][i] = 0;
        }
    }
    const int startZ = blockRegionFront - paddingStart < filterSize ? 0
                        : 1 + (blockRegionFront - paddingStart -filterSize) / moduleStride;
    const int endZ = MIN(numModulesZ, 1 + (blockRegionFront + 3 - paddingStart) / moduleStride);
    const int startY = blockRegionTop - paddingStart < filterSize ? 0
                        : 1 + (blockRegionTop - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesY, 1 + (blockRegionTop + 3 - paddingStart) / moduleStride);
    const int startX = blockRegionLeft - paddingStart < filterSize ? 0
                        : 1 + (blockRegionLeft - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockRegionLeft + 3 - paddingStart) / moduleStride);
    
    float* shilterLoad = &shFilters[threadIdx.y][threadIdx.x];
    float* shHidActLoad = &shHidActs[loadY][loadX];
    for (int mz = startZ; mz  < endZ; mz++){
        const int moduleFront = paddingStart + mz * moduleStride;
        const int pxInModuleZ = pxZ - moduleFront;

        for (int my = startY; my < endY; my++) {
            const int moduleTop = paddingStart + my * moduleStride;
            const int pxInModuleY = pxY - moduleTop;

            for (int mx = startX; mx < endX; mx++) {
                const int moduleIdx = mz * numModulesX * numModulesY + my * numModulesX + mx;
                const int moduleLeft = paddingStart + mx * moduleStride;
                const int pxInModuleX = pxX - moduleLeft;

                const bool isPxInModule = pxInModuleZ >= 0 && pxInModuleZ < filterSize && pxInModuleY >= 0 && pxInModuleY < filterSize && pxInModuleX >= 0 && pxInModuleX < filterSize;
                const int pxIdxInModule = pxInModuleZ * filterSize * filterSize + pxInModuleY * filterSize + pxInModuleX;

                for (int f = 0; f < numFilters; f += 16) { // multiply with 16 filters at a time
                    // Now the threads split up into half-warps, and each half-warp decides if it's interested.
                    const float* hLoad = &hidActs[(moduleIdx + f * numModules) * numImages];
                    #pragma unroll
                    for (int i = 0; i < 2 * 16; i += 32) { // IMAGES
                        if (!true || blockCaseIdx + i + loadX < numImages) {
                            #pragma unroll
                            for (int j = 0; j < 16; j += 8) { // load 16 rows of 2*16 cols, 8 * 32 elements at a time.
                                shHidActLoad[j * 16 * 2 + i] = hLoad[j * numModules * numImages + i];
                            }
                        } else {
                            #pragma unroll
                            for (int j = 0; j < 16; j += 8) { // load 16 rows of 2*16 cols, 8 * 32 elements at a time.
                                shHidActLoad[j * 16 * 2 + i] = 0;
                            }
                        }
                    }
                    
                    if (isPxInImg && isPxInModule) {
                        // This half-warp is interested, so it's going to load the weights from this module to its pixel.
                        // Not fully coalesced read :(
                        // But taking out this read entirely only reduces the runtime by ~2.8%, so it isn't costing me much.
                        const float* fLoad = true ? &filters[pxIdxInModule * numFilters + f]
                                                  : &filters[(moduleIdx * 1 * filterPixels + pxIdxInModule) * numFilters + f];
                        #pragma unroll
                        for (int c = 0; c < 1; c++) {
                            shilterLoad[c * 16 * (16 + 1)] = fLoad[c * filterPixels * numFilters];
                        }

                        
                    }

                    __syncthreads();
                    // Do some actual computation
                    if (isPxInImg && isPxInModule) {
                        #pragma unroll
                        for (int c = 0; c < 1; c++) {
                            #pragma unroll
                            for (int w = 0; w < 16; w++) {
                                #pragma unroll
                                for (int i = 0; i < 2; i++) {
                                    prod[c][i] += shFilters[threadIdx.y + c * 16][w] * shHidActs[w][threadIdx.x + i * 16];
                                }
                            }
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
    // Not fully coalesced write :(... shmem (and fully coalesced) version is actually slightly slower, though
    if (isPxInImg) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            if (!true || blockCaseIdx + threadIdx.x + i * 16 < numImages) {
                #pragma unroll
                for (int c = 0; c < 1; c++) {
                    targets[c * imgPixels * numImages + i * 16] = prod[c][i];
                }
            }
        }
    }
}

__global__ void kConvolve_forward_c(float* targets, float* images, float* filters,
                                       const int numImages, const int numFilters,
                                       const int imgSizeZ, const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesZ, const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups) {
    __shared__ float shFilters[4*2][4 * 8]; // pre-load 4 pixels from 4*8 filters
    __shared__ float shImages[4*2][32 * 1]; // pre-load 4 pixels from 32*1 images
    const int imgPixels = imgSizeZ * imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = numFilters / (4*8);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = 8 * 4 * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    const int numModules = numModulesX * numModulesY * numModulesZ;
    const int blockColorIdx = numFilterColors * blockGroupIdx;

    const int tidx = threadIdx.y * 32 + threadIdx.x;

    const int imgLoadModPosZ = paddingStart + (moduleIdx / (numModulesX * numModulesY)) * moduleStride;
    const int imgLoadModPosY = paddingStart + ((moduleIdx / numModulesX) % numModulesY) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (4 * 8);
    const int shFilterLoadX = tidx % (4 * 8);
    const int myImgIdx = blockIdx.x * 32 * 1 + threadIdx.x;

    images += blockColorIdx * imgPixels * imgStride + myImgIdx;
    filters +=blockFilterIdx
            + shFilterLoadY * numFilters + shFilterLoadX;
    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y) * numImages * numModules
            + myImgIdx;

    float prod[8][1];
    #pragma unroll
    for(int f = 0; f < 8; f++) {
        #pragma unroll
        for(int g = 0; g < 1; g++) {
            prod[f][g] = 0;
        }
    }
//    __shared__ int imgPos[]
    for (int oc = 0; oc < numFilterColors; oc += 2) { // oc stands for outer color (loop)
        for (int p = 0; p < filterPixels; p += 4) {
            /*
             * Load 4 pixels from 4*8 filters
             */
            if (shFilterLoadY < 4) {
                #pragma unroll
                for (int p2 = 0; p2 < 4; p2 += 32/8) {
                    if (p + p2 + shFilterLoadY < filterPixels) {
                        #pragma unroll
                        for (int c = 0; c < 2; c++) {
                            shFilters[shFilterLoadY + p2 + c * 4][shFilterLoadX] = filters[((oc+c) * filterPixels + p + p2) * numFilters];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < 2; c++) {
                            shFilters[shFilterLoadY + p2 + c * 4][shFilterLoadX] = 0;
                        }
                    }
                }
            }

            /*
             * Load 4 pixels from 32*1 images
             */
            const int pixIdx = p + threadIdx.y;
            if (pixIdx < filterPixels) {
                const int x = imgLoadModPosX + pixIdx % filterSize;
                const int y = imgLoadModPosY + (pixIdx / filterSize) % filterSize;
                const int z = imgLoadModPosZ + pixIdx / (filterSize * filterSize);
                if (z >= 0 && z < imgSizeZ && y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
                    float* m = &images[imgStride * (oc * imgPixels + z * imgSizeX * imgSizeY + y * imgSizeX + x)];
                    #pragma unroll
                    for (int i = 0; i < 1; i++) {
                        if (!true || myImgIdx + i * 32 < numImages) {
                            #pragma unroll
                            for (int c = 0; c < 2; c++) {
                                shImages[threadIdx.y + c * 4][threadIdx.x + i * 32] = m[c * imgStride * imgPixels + i * 32];
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < 2; c++) {
                                shImages[threadIdx.y + c * 4][threadIdx.x + i * 32] = 0;
                            }
                        }
                    }
                } else { // Padding
                    #pragma unroll
                    for (int i = 0; i < 1; i++) {
                        #pragma unroll
                        for (int c = 0; c < 2; c++) {
                            shImages[threadIdx.y + c * 4][threadIdx.x + i * 32] = 0;
                        }
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < 4*2; i++) {
                #pragma unroll
                for(int f = 0; f < 8; f++) {
                    #pragma unroll
                    for(int g = 0; g < 1; g++) {
                        prod[f][g] += shImages[i][g * 32 + threadIdx.x] * shFilters[i][threadIdx.y + f * 4];
                    }
                }

            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int g = 0; g < 1; g++) {
        if (!true || myImgIdx + g * 32 < numImages) {
            #pragma unroll
            for (int f = 0; f < 8; f++) {
                targets[g * 32 + f * 4 * numImages * numModules] =  prod[f][g];
            }
        }
    }
}

__global__ void kConvolve_weight_c(float* targets, float* images, float* hidActs,
                                       const int numImages, const int numFilters,
                                       const int numModulesZ, const int numModulesY, const int numModulesX,
                                       const int imgSizeZ, const int imgSizeY, const int imgSizeX, const int filterSize,
                                       const int paddingStart, const int moduleStride, const int imgStride,
                                       const int numImgColors, const int numGroups, const int partialSum,
                                       const float scaleOutputs) {
    __shared__ float shImages[8 * 8][32]; // preload 32 cases of 4 * pixelsPerThread pixels
    __shared__ float shHidActs[2 * 16][32 + 1]; // preload 32 cases of 32 hidacts

    const int tidx = 16 * threadIdx.y + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;

    const int filterPixels = filterSize * filterSize * filterSize;
    const int imgPixels = imgSizeZ * imgSizeY * imgSizeX;

    const int numFilterBlocks = numFilters / (16 * 2);
    const int outputModuleIdx = blockIdx.x / numFilterBlocks;
    const int moduleIdx = partialSum * outputModuleIdx;
    const int blockFilterIdx = 2 * 16 * (blockIdx.x % numFilterBlocks);
    const int numModules = numModulesZ * numModulesY * numModulesX;
    
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;
    const int numFilterColors = numImgColors / numGroups;
    
    const int blockPixelOffset = (blockIdx.y / (numFilterColors/8)) * 8;
    const int filterColorIdx = (blockIdx.y % (numFilterColors/8)) * 8;
    const int imgColorIdx = filterColorIdx + blockGroupIdx * numFilterColors;

    images += imgColorIdx * imgPixels * imgStride + loadX;

    hidActs += moduleIdx * numImages
            + blockFilterIdx * numImages * numModules
            + loadY * numImages * numModules
            + loadX;
    
    targets += outputModuleIdx * numFilters * filterPixels * numFilterColors
            + filterColorIdx * filterPixels * numFilters
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + threadIdx.y * numFilters + threadIdx.x;

    float* shHidActLoad = &shHidActs[loadY][loadX];
    float* shImgLoad = &shImages[loadY][loadX];
    float prod[8][2];
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        #pragma unroll
        for (int f = 0; f < 2; f++) {
            prod[c][f] = 0;
        }
    }
    // This avoids doing a division in an inner loop
    __shared__ int pxDivs[8];
    if (tidx < 8) {
        pxDivs[tidx] = (((blockPixelOffset + tidx) / (filterSize * filterSize)) << 16) + (((blockPixelOffset + tidx) / filterSize) % filterSize << 8) + ((blockPixelOffset + tidx) % filterSize);
    }
    __syncthreads();
    for (int m = moduleIdx; m < moduleIdx + partialSum; m++) {
        const int imgLoadModPosZ = paddingStart + (m / (numModulesY * numModulesX)) * moduleStride;
        const int imgLoadModPosY = paddingStart + ((m / numModulesX) % numModulesY) * moduleStride;
        const int imgLoadModPosX = paddingStart + (m % numModulesX) * moduleStride;
        for (int caseIdx = 0; caseIdx < numImages; caseIdx += 32) {
            if (loadY < 8) {
                /*
                 * As long as 4 * 32 is divisible by 32 this will loop the right
                 * number of times.
                 *
                 * This will load some images from filter pixels that don't exist (it'll set those to 0),
                 * but the code does not produce any output for those pixels (see last lines).
                 */
    //            #pragma unroll
                for (int y = 0; y < 8; y += (16 * 8) / 32) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if (8 % (16 * 8 / 32) == 0 || y + loadY < 8) {
                        const int pxIdx = loadY + y; // pixel idx in filter

                        if (pxIdx + blockPixelOffset < filterPixels && (!true || caseIdx + loadX < numImages)) {
                            const int pxZ = imgLoadModPosZ + HI24(pxDivs[pxIdx]);
                            const int pxY = imgLoadModPosY + MI16(pxDivs[pxIdx]);//pxIdx / filterSize; // pixel x,y coords in image
                            const int pxX = imgLoadModPosX + LO8(pxDivs[pxIdx]);
                            if (pxZ >= 0 && pxZ < imgSizeZ && pxY >= 0 && pxY < imgSizeY && pxX >= 0 && pxX < imgSizeX) {
                                const int pixIdx = (pxZ * imgSizeX * imgSizeY + pxY * imgSizeX + pxX) * imgStride; // pixel idx in image
                                #pragma unroll
                                for (int c = 0; c < 8; c++) {
                                    shImgLoad[(y + c * 8) * 32] = images[caseIdx + c * imgPixels * imgStride + pixIdx];
                                }
                            } else {
                                #pragma unroll
                                for (int c = 0; c < 8; c++) {
                                    shImgLoad[(y + c * 8) * 32] = 0;
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < 8; c++) {
                                shImgLoad[(y + c * 8) * 32] = 0;
                            }
                        }
                    }
                }
            }
            if (loadY < 16 * 2 && (!true || caseIdx + loadX < numImages)) {
                #pragma unroll
                for (int y = 0; y < 16 * 2; y += (16 * 8) / 32) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if ((16 * 2) % (16 * 8 / 32) == 0 || y + loadY < 16 * 2) {
                        shHidActLoad[y * (32 + 1)] = hidActs[caseIdx + y * numImages * numModules];
                    }
                }
            }

            __syncthreads();

            #pragma unroll
            for (int c = 0; c < 8; c++) {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    #pragma unroll
                    for (int f = 0; f < 2; f++) {
                        prod[c][f] += shImages[threadIdx.y + c * 8][i] * shHidActs[threadIdx.x + f * 16][i];
                    }
                }
            }
            __syncthreads();
        }
        hidActs += numImages;
    }
    if (blockPixelOffset + threadIdx.y < filterPixels) {
        #pragma unroll
        for (int f = 0; f < 2; f++) {
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                targets[c * filterPixels * numFilters + f * 16] = scaleOutputs * prod[c][f];
            }
        }
    }
}

__global__ void kConvolve_backward_c(float* targets, const float* hidActs, const float* filters, 
                                          const int numModulesZ, const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
                                          const int filterSize, const int imgSizeZ, const int imgSizeY, const int imgSizeX, const int paddingStart, const int moduleStride,
                                          const int numImgColors, const int numGroups) {
    __shared__ float shFilters[4*4][16 + 1]; // TODO: perhaps reconsider this 16
    __shared__ float shHidActs[16][32*1];

    const int numImgBlocks = DIVUP(numImages,32*1);
    const int blockCaseIdx = (blockIdx.x % numImgBlocks) * 32*1;
    
    const int imgColorIdx = (blockIdx.x / numImgBlocks) * 4*4; // color idx globally
    const int numFilterColors = numImgColors / numGroups;
    const int blockGroupIdx = imgColorIdx / numFilterColors;
    const int filterColorIdx = imgColorIdx % numFilterColors; // color idx within group
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockFilterIdx = blockGroupIdx * numFiltersPerGroup;

    const int blockPixelIdx = blockIdx.y;
    const int blockPixelIdxX = blockPixelIdx % imgSizeX;
    const int blockPixelIdxY = (blockPixelIdx / imgSizeX) % imgSizeY;
    const int blockPixelIdxZ = blockPixelIdx / (imgSizeX * imgSizeY);

    const int filterPixels = filterSize * filterSize * filterSize;
    const int imgPixels = imgSizeZ * imgSizeY * imgSizeX;
    const int tidx = threadIdx.y * 32 + threadIdx.x;
    const int hidActLoadY = tidx / 32, hidActLoadX = tidx % 32;
    const int filtersLoadY = tidx / 16, filtersLoadX = tidx % 16;
    const int numModules = numModulesZ * numModulesY * numModulesX;

    hidActs += blockCaseIdx + (blockFilterIdx + hidActLoadY) * numImages * numModules + hidActLoadX;
    filters += blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;
    targets += (imgColorIdx + threadIdx.y) * imgPixels * numImages + blockPixelIdx * numImages + blockCaseIdx + threadIdx.x;

    float prod[4][1];
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        #pragma unroll
        for (int i = 0; i < 1; i++) {
            prod[c][i] = 0;
        }
    }
    const int startZ = blockPixelIdxZ - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxZ - paddingStart - filterSize) / moduleStride;
    const int endZ = MIN(numModulesZ, 1 + (blockPixelIdxZ - paddingStart) / moduleStride); 
    const int startY = blockPixelIdxY - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxY - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesY, 1 + (blockPixelIdxY - paddingStart) / moduleStride);
    const int startX = blockPixelIdxX - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxX - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockPixelIdxX - paddingStart) / moduleStride);

    float* shFilterLoad = &shFilters[filtersLoadY][filtersLoadX];
    float* shHidActLoad = &shHidActs[hidActLoadY][hidActLoadX];
    for (int mz = startZ; mz < endZ ; mz++){
        const int moduleFront = paddingStart + mz * moduleStride;
        const int pxInFilterZ = blockPixelIdxZ - moduleFront;

        for (int my = startY; my < endY; my++) {
            const int moduleTop = paddingStart + my * moduleStride;
            const int pxInFilterY = blockPixelIdxY - moduleTop;

            for (int mx = startX; mx < endX; mx++) {
                const int moduleIdx = mz * numModulesX * numModulesY + my * numModulesX + mx;
                const int moduleLeft = paddingStart + mx * moduleStride;
                const int pxInFilterX = blockPixelIdxX - moduleLeft;
                
                const int pxIdxInFilter = pxInFilterZ * filterSize * filterSize + pxInFilterY * filterSize + pxInFilterX;

                for (int f = 0; f < numFiltersPerGroup; f += 16) { // multiply with 16 filters at a time
                    const float* hLoad = &hidActs[(moduleIdx + f * numModules) * numImages];
                    #pragma unroll
                    for (int i = 0; i < 1 * 32; i += 32) {
                        if (!true || blockCaseIdx + hidActLoadX + i < numImages) {
                            #pragma unroll
                            for (int j = 0; j < 16; j += 32*4/32) { // load 16 rows of 1*16 cols, 8 * 32 elements at a time.
                                shHidActLoad[j * 32 * 1 + i] = hLoad[j * numModules * numImages + i];
                            }
                        } else {
                            #pragma unroll
                            for (int j = 0; j < 16; j += 32*4/32) { // load 16 rows of 1*16 cols, 8 * 32 elements at a time.
                                shHidActLoad[j * 32 * 1 + i] = 0;
                            }
                        }
                    }
                    const float* fLoad = true ? &filters[pxIdxInFilter * numFilters + f]
                                              : &filters[moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + f];
                    #pragma unroll
                    for (int i = 0; i < 4*4; i+= 32*4/16) {
                        if ((4*4) % (32*4/16) == 0 || i + filtersLoadY < 4*4) {
                            shFilterLoad[i * (16 + 1)] = fLoad[i * filterPixels * numFilters];
                        }
                    }
                    
                    __syncthreads();
                    // Do some actual computation
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        #pragma unroll
                        for (int w = 0; w < 16; w++) {
                            #pragma unroll
                            for (int i = 0; i < 1; i++) {
                                prod[c][i] += shFilters[c * 4 + threadIdx.y][w] * shHidActs[w][threadIdx.x + i * 32];
                            }
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 1; i++) {
        if (!true || blockCaseIdx + threadIdx.x + i * 32 < numImages) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                targets[c * 4 * imgPixels * numImages + i * 32] = prod[c][i];
            }
        }
    }
}

