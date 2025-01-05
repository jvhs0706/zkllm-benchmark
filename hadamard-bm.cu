#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include "timer.hpp"
#include <string>

using namespace std;

KERNEL void e_step_kernel(GLOBAL Fr_t* arr_in, GLOBAL Fr_t* arr_out, Fr_t u, uint n_in)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n_in) return;

    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;

    u = blstrs__scalar__Scalar_mont(u);
    arr_out[gid1] = blstrs__scalar__Scalar_mul(arr_in[gid], u); // u * arr_in[gid]
    arr_out[gid0] = blstrs__scalar__Scalar_sub(arr_in[gid], arr_out[gid1]); // (1 - u) * arr[gid]
}

FrTensor e_step(const FrTensor& arr_in, const Fr_t& u)
{
    auto n_in = arr_in.size;
    FrTensor arr_out(2 * arr_in.size);
    e_step_kernel<<<(n_in+FrNumThread-1)/FrNumThread,FrNumThread>>>(arr_in.gpu_data, arr_out.gpu_data, u, n_in);
    cudaDeviceSynchronize();
    return arr_out;
}

FrTensor e(const FrTensor& arr_in, const vector<Fr_t> us)
{
    if (us.empty()) return arr_in;

    else {
        FrTensor arr_out = e_step(arr_in, us.back());
        return e(arr_out, {us.begin(), us.end() - 1});
    }
}

FrTensor e(const vector<Fr_t> us)
{
    Fr_t one = {1, 0, 0, 0, 0, 0, 0, 0};
    FrTensor arr_in (1, &one);
    return e(arr_in, us);
}

Fr_t one = {1, 0, 0, 0, 0, 0, 0, 0};
Fr_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

int main(int argc, char *argv[])
{   
    string prefix_left = argv[1];
    string prefix_right = argv[2];
    uint num = stoi(argv[3]);
    uint dim = stoi(argv[4]);

    Timer timer;

    vector<FrTensor> lslices, rslices;
    vector<Fr_t> ress;

    

    auto u_num = random_vec(ceilLog2(num));
    auto v_num = random_vec(ceilLog2(num));
    auto u_dim = random_vec(ceilLog2(dim));
    auto v_dim = random_vec(ceilLog2(dim));

    
    vector<Fr_t> l_auto(num), r_auto(num);

    for (uint i = 0; i < num; ++ i)
    {
        auto matl = FrTensor::from_int_bin(prefix_left + to_string(i)+".bin");
        auto matr = FrTensor::from_int_bin(prefix_right + to_string(i)+".bin");
        auto matres = matl * matr;
        
        // using the u_dims so that they can just be added up together, so that the proof size can be reduced
        // this means the proof size does not scale linearly with the number of instances !!!!!!!
        timer.start();
        auto proof1 = hadamard_product_sumcheck(matl, matr, u_dim, v_dim);
        l_auto[i] = matl(v_dim);
        r_auto[i] = matr(v_dim);
        timer.stop();
    }

    FrTensor l(num, &l_auto[0]);
    FrTensor r(num, &r_auto[0]);
    timer.start();
    auto proof2 = hadamard_product_sumcheck(l, r, u_num, v_num);
    timer.stop();
    cout << "Hadamard sumcheck time: " << timer.getTotalTime() << " s" << endl;

    // get the last cuda error
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaGetLastError failed: " << cudaGetErrorString(cudaStatus) << endl;
        return 1;
    }

    return 0;
}