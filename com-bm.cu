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

FrTensor e(const vector<Fr_t>& us)
{
    Fr_t one = {1, 0, 0, 0, 0, 0, 0, 0};
    FrTensor arr_in (1, &one);
    return e(arr_in, us);
}

Fr_t one = {1, 0, 0, 0, 0, 0, 0, 0};
Fr_t zero = {0, 0, 0, 0, 0, 0, 0, 0};

// this is really stupid but i don't want to change wheels
FrTensor const_tensor(const Fr_t& a, uint n)
{
    FrTensor res(n);
    res *= zero;
    res += a;
    return res;
}

int main(int argc, char *argv[])
{   
    string prefix = argv[1];
    string prefix_com = argv[2];
    uint num = stoi(argv[3]);
    uint dim = stoi(argv[4]);
    uint generator_dim = stoi(argv[5]);

    if (dim % generator_dim != 0) {
        cerr << "dim must be divisible by generator_dim" << endl;
        return 1;
    }

    uint com_size = 0;

    auto gen = Commitment::random(generator_dim);
    for (uint i = 0; i < num; ++ i) {
        auto mat = FrTensor::from_int_bin(prefix + to_string(i)+".bin");
        auto com = gen.commit_int(mat);
        com.save(prefix_com + to_string(i) + ".bin");
        com_size += com.size;
    }

    Timer timer;

    vector <Fr_t> u_num = random_vec(ceilLog2(num));
    vector <Fr_t> u_dim = random_vec(ceilLog2(dim));

    auto e_vec = e(u_num).trunc(0, num);

    FrTensor final_mat(dim);
    G1TensorJacobian final_com(dim / generator_dim);

    final_mat *= {0, 0 ,0, 0, 0, 0, 0 ,0};
    final_com *= const_tensor({0, 0 ,0, 0, 0, 0, 0 ,0}, final_com.size);

    for (uint i = 0; i < num; ++ i)
    {
        auto mat = FrTensor::from_int_bin(prefix + to_string(i)+".bin");
        auto com = G1TensorJacobian(prefix_com + to_string(i) + ".bin");
        timer.start();
        mat += final_mat * e_vec(i);
        com += final_com * const_tensor(e_vec(i), final_com.size);
        timer.stop();
    }

    timer.start();
    gen.open(final_mat, final_com, u_dim);
    timer.stop();
    cout << "Commitment opening time: " << timer.getTotalTime() << " s" << endl;

    // get the last cuda error
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaGetLastError failed: " << cudaGetErrorString(cudaStatus) << endl;
        return 1;
    }

    return 0;
}