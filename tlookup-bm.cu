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

KERNEL void tlookup_inv_kernel_bm(Fr_t* in_data, Fr_t beta, Fr_t* out_data, uint N)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < N){ 
        out_data[tid] = blstrs__scalar__Scalar_unmont(
            blstrs__scalar__Scalar_inverse(
                blstrs__scalar__Scalar_mont(
                    blstrs__scalar__Scalar_add(in_data[tid], beta)
                )
            )
        );
    }
}

Fr_t one = {1, 0, 0, 0, 0, 0, 0, 0};
Fr_t zero = {0, 0, 0, 0, 0, 0, 0, 0};
Fr_t two = {2, 0, 0, 0, 0, 0, 0, 0};

int main(int argc, char *argv[])
{   
    string prefix = argv[1];
    uint num = stoi(argv[2]);
    uint dim = stoi(argv[3]);
    int low = stoi(argv[4]);
    uint len = stoi(argv[5]);

    Timer timer;

    vector<FrTensor> lslices, rslices;
    vector<Fr_t> ress;

    auto u_num = random_vec(ceilLog2(num));
    auto v_num = random_vec(ceilLog2(num));
    auto u_dim = random_vec(ceilLog2(dim));
    auto v_dim = random_vec(ceilLog2(dim));

    tLookupRange tl(low, len);

    FrTensor* m_ptr = nullptr; 

    vector<Fr_t> beta_vec = random_vec(1);
    auto& beta = beta_vec[0];

    for (uint i = 0; i < num; ++ i)
    {
        auto s = FrTensor::from_int_bin(prefix + to_string(i)+".bin");
        FrTensor a(s.size);
        tlookup_inv_kernel_bm<<<(s.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            s.gpu_data,
            beta,
            a.gpu_data,
            s.size
        );
        cudaDeviceSynchronize();
        if (m_ptr == nullptr) m_ptr = new FrTensor(tl.prep(s));
        else *m_ptr += tl.prep(s);

        s.save(prefix + to_string(i)+"_s.bin");
        a.save(prefix + to_string(i)+"_a.bin");
    }

    uint num_ = num;
    while (num_ > 1)
    {

        uint num_new = 1 << (ceilLog2(num_) - 1);
        Fr_t middle_term = {0, 0, 0, 0, 0, 0, 0, 0};
        Fr_t a_sum_0 = {0, 0, 0, 0, 0, 0, 0, 0};
        Fr_t a_sum_1 = {0, 0, 0, 0, 0, 0, 0, 0};
        auto e_vec = e({u_num.begin(), u_num.end() - 1});
        for (uint i = 0; i < num_new; ++ i)
        {   
            auto i0 = i;
            auto i1 = i + num_new;
            if (i1 < num_)
            {
                FrTensor s0(prefix + to_string(i0)+"_s.bin");
                FrTensor s1(prefix + to_string(i1)+"_s.bin");
                FrTensor a0(prefix + to_string(i0)+"_a.bin");
                FrTensor a1(prefix + to_string(i1)+"_a.bin");

                timer.start();
                a_sum_0 = a_sum_0 + a0.sum();
                a_sum_1 = a_sum_1 + (a1 - a0).sum();
                

                Fr_t middle_term_temp = (a0 * (s1 + beta) + a1 * (s0 + beta))(u_dim);
                middle_term = middle_term + middle_term_temp * e_vec(i);
                FrTensor s_new = s0 + (s1 - s0) * v_num.back();
                FrTensor a_new = a0 + (a1 - a0) * v_num.back();
                timer.stop();

                s_new.save(prefix + to_string(i)+"_s.bin");
                a_new.save(prefix + to_string(i)+"_a.bin");
            }
            else 
            {
                FrTensor s0(prefix + to_string(i0)+"_s.bin");
                FrTensor a0(prefix + to_string(i0)+"_a.bin");

                timer.start();
                a_sum_0 = a_sum_0 + a0.sum();
                a_sum_1 = a_sum_1 - a0.sum();
                

                Fr_t middle_term_temp = (a0 * beta)(u_dim);
                middle_term = middle_term + middle_term_temp * e_vec(i);
                FrTensor s_new = s0 - s0 * v_num.back();
                FrTensor a_new = a0 - a0 * v_num.back();
                timer.stop();

                s_new.save(prefix + to_string(i)+"_s.bin");
                a_new.save(prefix + to_string(i)+"_a.bin");
            }
            
        }
        timer.start();
        Polynomial p ({one, middle_term - two, two - middle_term});
        p *= Polynomial::eq(u_num.back());
        u_num.pop_back();
        v_num.pop_back();
        p += Polynomial({a_sum_0, a_sum_1});
        num_ = num_new;
        timer.stop();
    }

    // get the last cuda error
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaGetLastError failed: " << cudaGetErrorString(cudaStatus) << endl;
        return 1;
    }

    cout << "tlookup sumcheck time: " << timer.getTotalTime() << endl;

    delete m_ptr;

    return 0;
}

// THIS VERSION IS RE-WRITTEN AFTER THE CAMERA-READY
// IT'S BASED ON MY MEMORY

