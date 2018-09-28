__global__ void extractIndices(long long nnz, const long long *indices, 
    int *rowIndices, int *colIndices, int tasks_per_thread = 8)
{
    auto global_id = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;
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