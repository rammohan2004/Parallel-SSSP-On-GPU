#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
using namespace std;

#define INF 2147483647 // simply INT_MAX
#define BLOCK_SIZE 256 // you can change it.

// ============================================================================
// CUDA KERNELS
// ============================================================================

__global__ void init(int *d_tent, int*d_far_flag, int *d_near_flag, int *d_proc_flag, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N) {
        d_tent[id] = INF;
        d_far_flag[id] = 0;
        d_near_flag[id] = -1;
        d_proc_flag[id] = -1; 
    }
}

__global__ void lightKernel(int *d_offsets, int *d_neighs, int*d_weights, int *d_near,
    int near_size,int * d_tent, int *d_near_next, int *d_processed, int *d_far,
    int *d_counters, int *d_far_flag, int *d_near_flag, int *d_proc_flag,
    int active_delta, int cutoff, int pass, int round ) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < near_size) {
       
        int u = d_near[id];
        if(d_proc_flag[u] != round) {
            d_proc_flag[u] = round;
            int slot = atomicAdd(&d_counters[1], 1);
            d_processed[slot] = u;
        }
        int du = d_tent[u];
        for(int e = d_offsets[u] ; e < d_offsets[u+1]; e++) {
            int w = d_weights[e];
            if(w > active_delta) continue;
            int v = d_neighs[e];
            int nd = du + w;
            int old = atomicMin(&d_tent[v], nd);
            if(nd < old) {
                if(nd <= cutoff) {
                    if(atomicExch(&d_near_flag[v], pass) != pass) {
                        int slot = atomicAdd(&d_counters[0], 1);
                        d_near_next[slot] = v;
                    }
                } else {
                    if(atomicCAS(&d_far_flag[v], 0, 1) == 0) {
                        int slot = atomicAdd(&d_counters[2], 1);
                        d_far[slot] = v;
                    }
                }
            }
        }
    }
}

__global__ void heavyKernel(int *d_offsets, int *d_neighs, int *d_weights, int *d_processed,
    int proc_size, int *d_tent, int *d_far, int *d_counters, int * d_far_flag, int active_delta) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if(id < proc_size) {
           int u = d_processed[id], du = d_tent[u];
           for(int e = d_offsets[u] ; e < d_offsets[u+1]; e++) {
                int w = d_weights[e];
                if(w <= active_delta) continue;
                int v = d_neighs[e];
                int nd = du + w;
                int old = atomicMin(&d_tent[v], nd);
                if(nd < old) {
                    if(atomicCAS(&d_far_flag[v], 0, 1) == 0) {
                        int slot = atomicAdd(&d_counters[2], 1);
                        d_far[slot] = v;
                    }
                }
            }

        }
    }

__global__ void pruneKernel(int *d_far, int far_size, int *d_tent, int *d_far_tmp,
                            int *d_far_keys, int *d_counters, int last_cutoff) {
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < far_size) {
        int v = d_far[id], k = d_tent[v];
        if(k > last_cutoff) {
            int slot = atomicAdd(&d_counters[3], 1);
            d_far_tmp[slot] = v;
            d_far_keys[slot] = k;
        }
    }             
}



// ============================================================================
// SINGLE SOURCE DELTA-STEPPING DRIVER
// ============================================================================

void run_delta_stepping_single_source(
    /**/
    int delta_mode, int K, int N, int E, int src,
    int *d_offsets, int *d_neighs, int *d_weights,
    int * d_tent, int *d_near, int *d_near_next, int *d_processed,
    int *d_far, int *d_counters, int *d_far_keys, int *d_far_tmp,
    int *d_far_flag, int *d_near_flag, int *d_proc_flag, vector<int>&deltas
)
{
    /**/
    int NBLOCKS = (N + BLOCK_SIZE - 1) / (BLOCK_SIZE);
    init<<<NBLOCKS, BLOCK_SIZE>>>(d_tent, d_far_flag, d_near_flag, d_proc_flag, N);
    int near_size = 1;
    int zero = 0;
    cudaMemcpy(d_tent+src, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_near, &src, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_counters, 0, 4 * sizeof(int));
    
    int far_size  = 0, proc_size = 0;
    int active_delta = 0, cutoff = 0, pass = 0, round = 0;
    int adaptive = (delta_mode == -1);

    while(true) {

        round++;
        cudaMemset(d_counters+1, 0, sizeof(int));

        while(near_size > 0) {
            pass++;
            cudaMemset(d_counters , 0, sizeof(int));
            int LKBLOCKS = (near_size + BLOCK_SIZE-1)/ BLOCK_SIZE;
            lightKernel<<<LKBLOCKS, BLOCK_SIZE>>> (d_offsets, d_neighs, d_weights, d_near, near_size,
                                                d_tent, d_near_next, d_processed, d_far, d_counters, 
                                                d_far_flag, d_near_flag, d_proc_flag, active_delta,
                                                cutoff, pass, round);
            cudaMemcpy(&near_size, d_counters, sizeof(int), cudaMemcpyDeviceToHost);
            swap(d_near, d_near_next);
        }

        cudaMemcpy(&proc_size, d_counters + 1, sizeof(int), cudaMemcpyDeviceToHost);
        if(proc_size > 0) {
            int HKBLOCKS = (proc_size + BLOCK_SIZE-1)/ BLOCK_SIZE;
            heavyKernel<<<HKBLOCKS, BLOCK_SIZE>>> (d_offsets, d_neighs, d_weights, d_processed, proc_size,
                                                d_tent, d_far, d_counters, d_far_flag, active_delta);
        }
        cudaMemcpy(&far_size, d_counters + 2, sizeof(int), cudaMemcpyDeviceToHost);

        int last_cutoff = cutoff;
        cudaMemset(d_counters + 3, 0, sizeof(int));
        if(far_size > 0) {
            int PKBLOCKS = (far_size + BLOCK_SIZE-1)/ BLOCK_SIZE;
            pruneKernel<<<PKBLOCKS, BLOCK_SIZE>>> (d_far, far_size, d_tent, d_far_tmp, d_far_keys,
                                                d_counters,last_cutoff);
        }
        cudaMemcpy(&far_size, d_counters + 3, sizeof(int), cudaMemcpyDeviceToHost);
        swap(d_far, d_far_tmp);
        cudaMemcpy(d_counters+2, &far_size, sizeof(int), cudaMemcpyHostToDevice);

        if(far_size == 0) break;

        thrust::device_ptr<int> d_far_thrust(d_far);
        thrust::device_ptr<int> d_far_keys_thrust(d_far_keys);
        thrust::sort_by_key(d_far_keys_thrust, d_far_keys_thrust +far_size, d_far_thrust);

        int Dmin, Dcutoff;
        cudaMemcpy(&Dmin, d_far_keys, sizeof(int), cudaMemcpyDeviceToHost);
        if(adaptive == 0) {
            Dcutoff = min((long long)Dmin + delta_mode, (long long)INF);
            active_delta = delta_mode;
            if(active_delta == 0) {
                active_delta = 1;
            }
        } else {
            int i = min(K-1, far_size-1);
            cudaMemcpy(&Dcutoff, d_far_keys + i, sizeof(int), cudaMemcpyDeviceToHost);
            int D = Dcutoff-Dmin;
            deltas.push_back(D);
            active_delta = D;
            if(active_delta == 0) {
                active_delta = 1;
            }
        }

        int cnt = thrust::upper_bound(d_far_keys_thrust,d_far_keys_thrust + far_size, Dcutoff) - d_far_keys_thrust;
        cudaMemcpy(d_near, d_far, cnt*sizeof(int), cudaMemcpyDeviceToDevice);
        int remaining = far_size- cnt;
        if (remaining > 0)
            cudaMemcpy(d_far_tmp,d_far + cnt, remaining*sizeof(int), cudaMemcpyDeviceToDevice);
        swap(d_far, d_far_tmp);

        near_size = cnt;
        far_size  = remaining;
        cudaMemcpy(d_counters+2, &far_size, sizeof(int), cudaMemcpyHostToDevice);
        cutoff = Dcutoff;
    }

}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile.is_open())
    {
        std::cerr << "Error: Unable to open input file " << argv[1] << "\n";
        return 1;
    }

    std::ofstream outfile(argv[2]);
    if (!outfile.is_open())
    {
        std::cerr << "Error: Unable to open output file " << argv[2] << "\n";
        return 1;
    }

    int delta_mode, K, N, E, S_count;
    infile >> delta_mode >> K;
    infile >> N >> E >> S_count;

    std::vector<int> sources(S_count);
    for (int i = 0; i < S_count; ++i)
    {
        infile >> sources[i];
    }

    int *offsets = new int[ N + 1 ] { 0 };
    for (int i = 0; i <= N; ++i)
    {
        infile >> offsets[i];
    }

    int *neighs = new int[ E ] { 0 };
    for (int i = 0; i < E; ++i)
    {
        infile >> neighs[i];
    }

    int *weights = new int[ E ] { 0 };
    for (int i = 0; i < E; ++i)
    {
        infile >> weights[i];
    }

    infile.close();

    int *d_offsets = nullptr, *d_neighs = nullptr, *d_weights = nullptr;
    cudaMalloc(&d_offsets, (N + 1) * sizeof(int));
    cudaMalloc(&d_neighs, E * sizeof(int));
    cudaMalloc(&d_weights, E * sizeof(int));

    cudaMemcpy(d_offsets, offsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighs, neighs, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, E * sizeof(int), cudaMemcpyHostToDevice);

    int *h_tent = new int[ N ] { 0 }; // sssp distance arry. tent means tentative distance
    int *d_tent;
    cudaMalloc(&d_tent, N * sizeof(int));


    /*
    ToDo
    */
    int  *d_near, *d_near_next, *d_processed, *d_far, *d_counters;
    int *d_far_keys, *d_far_tmp, *d_far_flag, *d_near_flag, *d_proc_flag;
    cudaMalloc(&d_near, N * sizeof(int));
    cudaMalloc(&d_near_next, N * sizeof(int));
    cudaMalloc(&d_processed, N * sizeof(int));
    cudaMalloc(&d_far, N * sizeof(int));
    cudaMalloc(&d_far_keys, N * sizeof(int));
    cudaMalloc(&d_far_tmp, N * sizeof(int));
    cudaMalloc(&d_far_flag, N * sizeof(int));
    cudaMalloc(&d_near_flag, N * sizeof(int));
    cudaMalloc(&d_proc_flag, N * sizeof(int));
    cudaMalloc(&d_counters, 4 * sizeof(int));
    vector<int>deltas;

    for (int i = 0; i < S_count; ++i)
    {
        int source = sources[i];
        deltas.clear();
        outfile << source << "\n";

        run_delta_stepping_single_source(
            /*
            ToDo*/
            delta_mode, K, N, E, source,
            d_offsets, d_neighs, d_weights,
            d_tent, d_near, d_near_next, d_processed,
            d_far, d_counters, d_far_keys, d_far_tmp,
            d_far_flag, d_near_flag, d_proc_flag, deltas
        );

        cudaMemcpy(h_tent, d_tent, N * sizeof(int), cudaMemcpyDeviceToHost);
        for(int j = 0; j < deltas.size(); j++) {
            outfile << deltas[j] << "\n";
        }
        for (int v = 0; v < N; ++v)
        {
            outfile << h_tent[v] << "\n";
        }
    }

    delete[] offsets;
    delete[] neighs;
    delete[] weights;
    delete[] h_tent;
    cudaFree(d_near);
    cudaFree(d_near_next);
    cudaFree(d_far);
    cudaFree(d_counters);
    cudaFree(d_processed);
    cudaFree(d_far_keys);
    cudaFree(d_far_tmp);
    cudaFree(d_far_flag);
    cudaFree(d_near_flag);
    cudaFree(d_proc_flag);
    cudaFree(d_tent);
    cudaFree(d_offsets);
    cudaFree(d_neighs);
    cudaFree(d_weights);

    outfile.close();
    return 0;
}