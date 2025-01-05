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
    uint dim0 = stoi(argv[4]);
    uint dim1 = stoi(argv[5]);
    uint dim2 = stoi(argv[6]);

    Timer timer;

    vector<FrTensor> lslices, rslices;
    vector<Fr_t> ress;

    auto u0 = random_vec(ceilLog2(dim0));
    auto u1 = random_vec(ceilLog2(dim1));
    auto u2 = random_vec(ceilLog2(dim2));

    for (uint i = 0; i < num; ++ i)
    {
        auto matl = FrTensor::from_int_bin(prefix_left + to_string(i)+".bin");
        auto matr = FrTensor::from_int_bin(prefix_right + to_string(i)+".bin");
        auto matres = FrTensor::matmul(matl, matr, dim0, dim1, dim2);

        if (matl.size != dim0 * dim1 || matr.size != dim1 * dim2 || matres.size != dim0 * dim2)
        {
            cerr << "Matrix size error!" << endl;
            return 1;
        }

        timer.start();
        lslices.push_back(matl.partial_me(u0, dim1));
        rslices.push_back(matr.partial_me(u2, 1));
        ress.push_back(matres.multi_dim_me({u0, u2}, {dim0, dim2}));
        timer.stop();
    }

    auto L = catTensors(lslices);
    auto R = catTensors(rslices);
    FrTensor Res(ress.size(), & ress.front()); 

    auto unum = random_vec(ceilLog2(num));
    auto vnum = random_vec(ceilLog2(num));
    auto ubatch = random_vec(ceilLog2(dim0));
    auto claim = Res(unum);

    timer.start();
    vector<Polynomial> proof;
    zkip_stacked(claim, L, R, unum, u1, vnum, num, dim1, proof);
    timer.stop();
    cout << "Matmul sumcheck time: " << timer.getTotalTime() << " s" << endl;

    // get the last cuda error
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaGetLastError failed: " << cudaGetErrorString(cudaStatus) << endl;
        return 1;
    }

    return 0;
}