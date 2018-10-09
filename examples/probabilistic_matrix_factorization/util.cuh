__device__ int getGlobalId()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int getGlobalStride()
{
    return blockDim.x * gridDim.x;
}

__global__ void extractIndices(long long nnz, const long long *indices, 
    int *rowIndices, int *colIndices, int tasks_per_thread = 8)
{
    auto global_id = getGlobalId();
    auto stride = getGlobalStride();
    for (auto i = 0; i < tasks_per_thread; ++i)
    {
        if (global_id >= nnz)
            break;
        auto j = global_id * 2;
        rowIndices[global_id] = (int)indices[j];
        colIndices[global_id] = (int)indices[j + 1];
        global_id += stride;
    }
}

__global__ void initIndices(long long nnz, int *indices, 
    int tasks_per_thread = 8)
{
    auto global_id = getGlobalId();
    auto stride = getGlobalStride();
    for (auto i = 0; i < tasks_per_thread; ++i)
    {
        if (global_id >= nnz)
            break;
        indices[global_id] = global_id;
        global_id += stride;
    }
}

__global__ void resort(long long nnz, int *indices, const int *cols, 
    const float *vals, int *cols_cpy, float *vals_cpy, int tasks_per_thread = 8)
{
    auto global_id = getGlobalId();
    auto stride = getGlobalStride();
    for (auto i = 0; i < tasks_per_thread; ++i)
    {
        if (global_id >= nnz)
            break;
        auto id = indices[global_id];
        cols_cpy[global_id] = cols[id];
        vals_cpy[global_id] = vals[id];
        //printf("%d, %d, %d, %f, %f\n", id, cols[id], cols_cpy[global_id], vals_cpy[global_id], vals[id]);
        global_id += stride;
    }
}