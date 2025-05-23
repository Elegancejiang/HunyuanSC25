/*
 * (c) 2015 Virginia Polytechnic Institute & State University (Virginia Tech)
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, version 2.1
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License, version 2.1, for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *
 */

#ifndef _H_BB_EXCH
#define _H_BB_EXCH

#define CMP_SWP(t1, _a, _b, t2, _c, _d) \
    if (_a > _b)                        \
    {                                   \
        t1 _t = _a;                     \
        _a = _b;                        \
        _b = _t;                        \
        t2 _s = _c;                     \
        _c = _d;                        \
        _d = _s;                        \
    }
#define EQL_SWP(t1, _a, _b, t2, _c, _d) \
    if (_a != _b)                       \
    {                                   \
        t1 _t = _a;                     \
        _a = _b;                        \
        _b = _t;                        \
        t2 _s = _c;                     \
        _c = _d;                        \
        _d = _s;                        \
    }
#define SWP(t1, _a, _b, t2, _c, _d) \
    {                               \
        t1 _t = _a;                 \
        _a = _b;                    \
        _b = _t;                    \
        t2 _s = _c;                 \
        _c = _d;                    \
        _d = _s;                    \
    }
// Exchange intersection for 1 keys.
template <class K>
__device__ inline void exch_intxn(K &k0, int &v0, int mask, const int bit)
{
    K ex_k0, ex_k1;
    int ex_v0, ex_v1;
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff, k0, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor_sync(0xffffffff, v0, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    v0 = ex_v0;
}
// Exchange intersection for 2 keys.
template <class K>
__device__ inline void exch_intxn(K &k0, K &k1, int &v0, int &v1, int mask, const int bit)
{
    K ex_k0, ex_k1;
    int ex_v0, ex_v1;
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff, k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor_sync(0xffffffff, v1, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
}
// Exchange intersection for 4 keys.
template <class K>
__device__ inline void exch_intxn(K &k0, K &k1, K &k2, K &k3, int &v0, int &v1, int &v2, int &v3, int mask, const int bit)
{
    K ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if (bit)
        SWP(K, k0, k2, int, v0, v2);
    if (bit)
        SWP(K, k1, k3, int, v1, v3);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff, k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor_sync(0xffffffff, v1, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff, k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor_sync(0xffffffff, v3, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    if (bit)
        SWP(K, k0, k2, int, v0, v2);
    if (bit)
        SWP(K, k1, k3, int, v1, v3);
}
// Exchange intersection for 8 keys.
template <class K>
__device__ inline void exch_intxn(K &k0, K &k1, K &k2, K &k3, K &k4, K &k5, K &k6, K &k7, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int mask, const int bit)
{
    K ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if (bit)
        SWP(K, k0, k6, int, v0, v6);
    if (bit)
        SWP(K, k1, k7, int, v1, v7);
    if (bit)
        SWP(K, k2, k4, int, v2, v4);
    if (bit)
        SWP(K, k3, k5, int, v3, v5);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff, k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor_sync(0xffffffff, v1, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff, k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor_sync(0xffffffff, v3, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor_sync(0xffffffff, k5, mask);
    ex_v0 = v4;
    ex_v1 = __shfl_xor_sync(0xffffffff, v5, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k4 = ex_k0;
    k5 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v4 = ex_v0;
    v5 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor_sync(0xffffffff, k7, mask);
    ex_v0 = v6;
    ex_v1 = __shfl_xor_sync(0xffffffff, v7, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k6 = ex_k0;
    k7 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v6 = ex_v0;
    v7 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    if (bit)
        SWP(K, k0, k6, int, v0, v6);
    if (bit)
        SWP(K, k1, k7, int, v1, v7);
    if (bit)
        SWP(K, k2, k4, int, v2, v4);
    if (bit)
        SWP(K, k3, k5, int, v3, v5);
}
// Exchange intersection for 16 keys.
template <class K>
__device__ inline void exch_intxn(K &k0, K &k1, K &k2, K &k3, K &k4, K &k5, K &k6, K &k7, K &k8, K &k9, K &k10, K &k11, K &k12, K &k13, K &k14, K &k15, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int &v8, int &v9, int &v10, int &v11, int &v12, int &v13, int &v14, int &v15, int mask, const int bit)
{
    K ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if (bit)
        SWP(K, k0, k14, int, v0, v14);
    if (bit)
        SWP(K, k1, k15, int, v1, v15);
    if (bit)
        SWP(K, k2, k12, int, v2, v12);
    if (bit)
        SWP(K, k3, k13, int, v3, v13);
    if (bit)
        SWP(K, k4, k10, int, v4, v10);
    if (bit)
        SWP(K, k5, k11, int, v5, v11);
    if (bit)
        SWP(K, k6, k8, int, v6, v8);
    if (bit)
        SWP(K, k7, k9, int, v7, v9);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff, k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor_sync(0xffffffff, v1, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff, k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor_sync(0xffffffff, v3, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor_sync(0xffffffff, k5, mask);
    ex_v0 = v4;
    ex_v1 = __shfl_xor_sync(0xffffffff, v5, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k4 = ex_k0;
    k5 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v4 = ex_v0;
    v5 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor_sync(0xffffffff, k7, mask);
    ex_v0 = v6;
    ex_v1 = __shfl_xor_sync(0xffffffff, v7, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k6 = ex_k0;
    k7 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v6 = ex_v0;
    v7 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k8;
    ex_k1 = __shfl_xor_sync(0xffffffff, k9, mask);
    ex_v0 = v8;
    ex_v1 = __shfl_xor_sync(0xffffffff, v9, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k8 = ex_k0;
    k9 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v8 = ex_v0;
    v9 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k10;
    ex_k1 = __shfl_xor_sync(0xffffffff, k11, mask);
    ex_v0 = v10;
    ex_v1 = __shfl_xor_sync(0xffffffff, v11, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k10 = ex_k0;
    k11 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v10 = ex_v0;
    v11 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k12;
    ex_k1 = __shfl_xor_sync(0xffffffff, k13, mask);
    ex_v0 = v12;
    ex_v1 = __shfl_xor_sync(0xffffffff, v13, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k12 = ex_k0;
    k13 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v12 = ex_v0;
    v13 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k14;
    ex_k1 = __shfl_xor_sync(0xffffffff, k15, mask);
    ex_v0 = v14;
    ex_v1 = __shfl_xor_sync(0xffffffff, v15, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k14 = ex_k0;
    k15 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v14 = ex_v0;
    v15 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    if (bit)
        SWP(K, k0, k14, int, v0, v14);
    if (bit)
        SWP(K, k1, k15, int, v1, v15);
    if (bit)
        SWP(K, k2, k12, int, v2, v12);
    if (bit)
        SWP(K, k3, k13, int, v3, v13);
    if (bit)
        SWP(K, k4, k10, int, v4, v10);
    if (bit)
        SWP(K, k5, k11, int, v5, v11);
    if (bit)
        SWP(K, k6, k8, int, v6, v8);
    if (bit)
        SWP(K, k7, k9, int, v7, v9);
}
// Exchange intersection for 32 keys.
template <class K>
__device__ inline void exch_intxn(K &k0, K &k1, K &k2, K &k3, K &k4, K &k5, K &k6, K &k7, K &k8, K &k9, K &k10, K &k11, K &k12, K &k13, K &k14, K &k15, K &k16, K &k17, K &k18, K &k19, K &k20, K &k21, K &k22, K &k23, K &k24, K &k25, K &k26, K &k27, K &k28, K &k29, K &k30, K &k31, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int &v8, int &v9, int &v10, int &v11, int &v12, int &v13, int &v14, int &v15, int &v16, int &v17, int &v18, int &v19, int &v20, int &v21, int &v22, int &v23, int &v24, int &v25, int &v26, int &v27, int &v28, int &v29, int &v30, int &v31, int mask, const int bit)
{
    K ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if (bit)
        SWP(K, k0, k30, int, v0, v30);
    if (bit)
        SWP(K, k1, k31, int, v1, v31);
    if (bit)
        SWP(K, k2, k28, int, v2, v28);
    if (bit)
        SWP(K, k3, k29, int, v3, v29);
    if (bit)
        SWP(K, k4, k26, int, v4, v26);
    if (bit)
        SWP(K, k5, k27, int, v5, v27);
    if (bit)
        SWP(K, k6, k24, int, v6, v24);
    if (bit)
        SWP(K, k7, k25, int, v7, v25);
    if (bit)
        SWP(K, k8, k22, int, v8, v22);
    if (bit)
        SWP(K, k9, k23, int, v9, v23);
    if (bit)
        SWP(K, k10, k20, int, v10, v20);
    if (bit)
        SWP(K, k11, k21, int, v11, v21);
    if (bit)
        SWP(K, k12, k18, int, v12, v18);
    if (bit)
        SWP(K, k13, k19, int, v13, v19);
    if (bit)
        SWP(K, k14, k16, int, v14, v16);
    if (bit)
        SWP(K, k15, k17, int, v15, v17);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff, k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor_sync(0xffffffff, v1, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff, k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor_sync(0xffffffff, v3, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor_sync(0xffffffff, k5, mask);
    ex_v0 = v4;
    ex_v1 = __shfl_xor_sync(0xffffffff, v5, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k4 = ex_k0;
    k5 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v4 = ex_v0;
    v5 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor_sync(0xffffffff, k7, mask);
    ex_v0 = v6;
    ex_v1 = __shfl_xor_sync(0xffffffff, v7, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k6 = ex_k0;
    k7 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v6 = ex_v0;
    v7 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k8;
    ex_k1 = __shfl_xor_sync(0xffffffff, k9, mask);
    ex_v0 = v8;
    ex_v1 = __shfl_xor_sync(0xffffffff, v9, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k8 = ex_k0;
    k9 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v8 = ex_v0;
    v9 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k10;
    ex_k1 = __shfl_xor_sync(0xffffffff, k11, mask);
    ex_v0 = v10;
    ex_v1 = __shfl_xor_sync(0xffffffff, v11, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k10 = ex_k0;
    k11 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v10 = ex_v0;
    v11 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k12;
    ex_k1 = __shfl_xor_sync(0xffffffff, k13, mask);
    ex_v0 = v12;
    ex_v1 = __shfl_xor_sync(0xffffffff, v13, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k12 = ex_k0;
    k13 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v12 = ex_v0;
    v13 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k14;
    ex_k1 = __shfl_xor_sync(0xffffffff, k15, mask);
    ex_v0 = v14;
    ex_v1 = __shfl_xor_sync(0xffffffff, v15, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k14 = ex_k0;
    k15 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v14 = ex_v0;
    v15 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k16;
    ex_k1 = __shfl_xor_sync(0xffffffff, k17, mask);
    ex_v0 = v16;
    ex_v1 = __shfl_xor_sync(0xffffffff, v17, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k16 = ex_k0;
    k17 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v16 = ex_v0;
    v17 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k18;
    ex_k1 = __shfl_xor_sync(0xffffffff, k19, mask);
    ex_v0 = v18;
    ex_v1 = __shfl_xor_sync(0xffffffff, v19, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k18 = ex_k0;
    k19 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v18 = ex_v0;
    v19 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k20;
    ex_k1 = __shfl_xor_sync(0xffffffff, k21, mask);
    ex_v0 = v20;
    ex_v1 = __shfl_xor_sync(0xffffffff, v21, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k20 = ex_k0;
    k21 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v20 = ex_v0;
    v21 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k22;
    ex_k1 = __shfl_xor_sync(0xffffffff, k23, mask);
    ex_v0 = v22;
    ex_v1 = __shfl_xor_sync(0xffffffff, v23, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k22 = ex_k0;
    k23 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v22 = ex_v0;
    v23 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k24;
    ex_k1 = __shfl_xor_sync(0xffffffff, k25, mask);
    ex_v0 = v24;
    ex_v1 = __shfl_xor_sync(0xffffffff, v25, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k24 = ex_k0;
    k25 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v24 = ex_v0;
    v25 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k26;
    ex_k1 = __shfl_xor_sync(0xffffffff, k27, mask);
    ex_v0 = v26;
    ex_v1 = __shfl_xor_sync(0xffffffff, v27, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k26 = ex_k0;
    k27 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v26 = ex_v0;
    v27 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k28;
    ex_k1 = __shfl_xor_sync(0xffffffff, k29, mask);
    ex_v0 = v28;
    ex_v1 = __shfl_xor_sync(0xffffffff, v29, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k28 = ex_k0;
    k29 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v28 = ex_v0;
    v29 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k30;
    ex_k1 = __shfl_xor_sync(0xffffffff, k31, mask);
    ex_v0 = v30;
    ex_v1 = __shfl_xor_sync(0xffffffff, v31, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k30 = ex_k0;
    k31 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v30 = ex_v0;
    v31 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    if (bit)
        SWP(K, k0, k30, int, v0, v30);
    if (bit)
        SWP(K, k1, k31, int, v1, v31);
    if (bit)
        SWP(K, k2, k28, int, v2, v28);
    if (bit)
        SWP(K, k3, k29, int, v3, v29);
    if (bit)
        SWP(K, k4, k26, int, v4, v26);
    if (bit)
        SWP(K, k5, k27, int, v5, v27);
    if (bit)
        SWP(K, k6, k24, int, v6, v24);
    if (bit)
        SWP(K, k7, k25, int, v7, v25);
    if (bit)
        SWP(K, k8, k22, int, v8, v22);
    if (bit)
        SWP(K, k9, k23, int, v9, v23);
    if (bit)
        SWP(K, k10, k20, int, v10, v20);
    if (bit)
        SWP(K, k11, k21, int, v11, v21);
    if (bit)
        SWP(K, k12, k18, int, v12, v18);
    if (bit)
        SWP(K, k13, k19, int, v13, v19);
    if (bit)
        SWP(K, k14, k16, int, v14, v16);
    if (bit)
        SWP(K, k15, k17, int, v15, v17);
}
// Exchange parallel for 1 keys.
template <class K>
__device__ inline void exch_paral(K &k0, int &v0, int mask, const int bit)
{
    K ex_k0, ex_k1;
    int ex_v0, ex_v1;
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff, k0, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor_sync(0xffffffff, v0, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    v0 = ex_v0;
}
// Exchange parallel for 2 keys.
template <class K>
__device__ inline void exch_paral(K &k0, K &k1, int &v0, int &v1, int mask, const int bit)
{
    K ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if (bit)
        SWP(K, k0, k1, int, v0, v1);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff, k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor_sync(0xffffffff, v1, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    if (bit)
        SWP(K, k0, k1, int, v0, v1);
}
// Exchange parallel for 4 keys.
template <class K>
__device__ inline void exch_paral(K &k0, K &k1, K &k2, K &k3, int &v0, int &v1, int &v2, int &v3, int mask, const int bit)
{
    K ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if (bit)
        SWP(K, k0, k1, int, v0, v1);
    if (bit)
        SWP(K, k2, k3, int, v2, v3);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff, k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor_sync(0xffffffff, v1, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff, k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor_sync(0xffffffff, v3, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    if (bit)
        SWP(K, k0, k1, int, v0, v1);
    if (bit)
        SWP(K, k2, k3, int, v2, v3);
}
// Exchange parallel for 8 keys.
template <class K>
__device__ inline void exch_paral(K &k0, K &k1, K &k2, K &k3, K &k4, K &k5, K &k6, K &k7, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int mask, const int bit)
{
    K ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if (bit)
        SWP(K, k0, k1, int, v0, v1);
    if (bit)
        SWP(K, k2, k3, int, v2, v3);
    if (bit)
        SWP(K, k4, k5, int, v4, v5);
    if (bit)
        SWP(K, k6, k7, int, v6, v7);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff, k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor_sync(0xffffffff, v1, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff, k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor_sync(0xffffffff, v3, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor_sync(0xffffffff, k5, mask);
    ex_v0 = v4;
    ex_v1 = __shfl_xor_sync(0xffffffff, v5, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k4 = ex_k0;
    k5 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v4 = ex_v0;
    v5 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor_sync(0xffffffff, k7, mask);
    ex_v0 = v6;
    ex_v1 = __shfl_xor_sync(0xffffffff, v7, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k6 = ex_k0;
    k7 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v6 = ex_v0;
    v7 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    if (bit)
        SWP(K, k0, k1, int, v0, v1);
    if (bit)
        SWP(K, k2, k3, int, v2, v3);
    if (bit)
        SWP(K, k4, k5, int, v4, v5);
    if (bit)
        SWP(K, k6, k7, int, v6, v7);
}
// Exchange parallel for 16 keys.
template <class K>
__device__ inline void exch_paral(K &k0, K &k1, K &k2, K &k3, K &k4, K &k5, K &k6, K &k7, K &k8, K &k9, K &k10, K &k11, K &k12, K &k13, K &k14, K &k15, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int &v8, int &v9, int &v10, int &v11, int &v12, int &v13, int &v14, int &v15, int mask, const int bit)
{
    K ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if (bit)
        SWP(K, k0, k1, int, v0, v1);
    if (bit)
        SWP(K, k2, k3, int, v2, v3);
    if (bit)
        SWP(K, k4, k5, int, v4, v5);
    if (bit)
        SWP(K, k6, k7, int, v6, v7);
    if (bit)
        SWP(K, k8, k9, int, v8, v9);
    if (bit)
        SWP(K, k10, k11, int, v10, v11);
    if (bit)
        SWP(K, k12, k13, int, v12, v13);
    if (bit)
        SWP(K, k14, k15, int, v14, v15);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff, k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor_sync(0xffffffff, v1, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff, k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor_sync(0xffffffff, v3, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor_sync(0xffffffff, k5, mask);
    ex_v0 = v4;
    ex_v1 = __shfl_xor_sync(0xffffffff, v5, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k4 = ex_k0;
    k5 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v4 = ex_v0;
    v5 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor_sync(0xffffffff, k7, mask);
    ex_v0 = v6;
    ex_v1 = __shfl_xor_sync(0xffffffff, v7, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k6 = ex_k0;
    k7 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v6 = ex_v0;
    v7 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k8;
    ex_k1 = __shfl_xor_sync(0xffffffff, k9, mask);
    ex_v0 = v8;
    ex_v1 = __shfl_xor_sync(0xffffffff, v9, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k8 = ex_k0;
    k9 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v8 = ex_v0;
    v9 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k10;
    ex_k1 = __shfl_xor_sync(0xffffffff, k11, mask);
    ex_v0 = v10;
    ex_v1 = __shfl_xor_sync(0xffffffff, v11, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k10 = ex_k0;
    k11 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v10 = ex_v0;
    v11 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k12;
    ex_k1 = __shfl_xor_sync(0xffffffff, k13, mask);
    ex_v0 = v12;
    ex_v1 = __shfl_xor_sync(0xffffffff, v13, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k12 = ex_k0;
    k13 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v12 = ex_v0;
    v13 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k14;
    ex_k1 = __shfl_xor_sync(0xffffffff, k15, mask);
    ex_v0 = v14;
    ex_v1 = __shfl_xor_sync(0xffffffff, v15, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k14 = ex_k0;
    k15 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v14 = ex_v0;
    v15 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    if (bit)
        SWP(K, k0, k1, int, v0, v1);
    if (bit)
        SWP(K, k2, k3, int, v2, v3);
    if (bit)
        SWP(K, k4, k5, int, v4, v5);
    if (bit)
        SWP(K, k6, k7, int, v6, v7);
    if (bit)
        SWP(K, k8, k9, int, v8, v9);
    if (bit)
        SWP(K, k10, k11, int, v10, v11);
    if (bit)
        SWP(K, k12, k13, int, v12, v13);
    if (bit)
        SWP(K, k14, k15, int, v14, v15);
}
// Exchange parallel for 32 keys.
template <class K>
__device__ inline void exch_paral(K &k0, K &k1, K &k2, K &k3, K &k4, K &k5, K &k6, K &k7, K &k8, K &k9, K &k10, K &k11, K &k12, K &k13, K &k14, K &k15, K &k16, K &k17, K &k18, K &k19, K &k20, K &k21, K &k22, K &k23, K &k24, K &k25, K &k26, K &k27, K &k28, K &k29, K &k30, K &k31, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int &v8, int &v9, int &v10, int &v11, int &v12, int &v13, int &v14, int &v15, int &v16, int &v17, int &v18, int &v19, int &v20, int &v21, int &v22, int &v23, int &v24, int &v25, int &v26, int &v27, int &v28, int &v29, int &v30, int &v31, int mask, const int bit)
{
    K ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if (bit)
        SWP(K, k0, k1, int, v0, v1);
    if (bit)
        SWP(K, k2, k3, int, v2, v3);
    if (bit)
        SWP(K, k4, k5, int, v4, v5);
    if (bit)
        SWP(K, k6, k7, int, v6, v7);
    if (bit)
        SWP(K, k8, k9, int, v8, v9);
    if (bit)
        SWP(K, k10, k11, int, v10, v11);
    if (bit)
        SWP(K, k12, k13, int, v12, v13);
    if (bit)
        SWP(K, k14, k15, int, v14, v15);
    if (bit)
        SWP(K, k16, k17, int, v16, v17);
    if (bit)
        SWP(K, k18, k19, int, v18, v19);
    if (bit)
        SWP(K, k20, k21, int, v20, v21);
    if (bit)
        SWP(K, k22, k23, int, v22, v23);
    if (bit)
        SWP(K, k24, k25, int, v24, v25);
    if (bit)
        SWP(K, k26, k27, int, v26, v27);
    if (bit)
        SWP(K, k28, k29, int, v28, v29);
    if (bit)
        SWP(K, k30, k31, int, v30, v31);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff, k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor_sync(0xffffffff, v1, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff, k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor_sync(0xffffffff, v3, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor_sync(0xffffffff, k5, mask);
    ex_v0 = v4;
    ex_v1 = __shfl_xor_sync(0xffffffff, v5, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k4 = ex_k0;
    k5 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v4 = ex_v0;
    v5 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor_sync(0xffffffff, k7, mask);
    ex_v0 = v6;
    ex_v1 = __shfl_xor_sync(0xffffffff, v7, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k6 = ex_k0;
    k7 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v6 = ex_v0;
    v7 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k8;
    ex_k1 = __shfl_xor_sync(0xffffffff, k9, mask);
    ex_v0 = v8;
    ex_v1 = __shfl_xor_sync(0xffffffff, v9, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k8 = ex_k0;
    k9 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v8 = ex_v0;
    v9 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k10;
    ex_k1 = __shfl_xor_sync(0xffffffff, k11, mask);
    ex_v0 = v10;
    ex_v1 = __shfl_xor_sync(0xffffffff, v11, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k10 = ex_k0;
    k11 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v10 = ex_v0;
    v11 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k12;
    ex_k1 = __shfl_xor_sync(0xffffffff, k13, mask);
    ex_v0 = v12;
    ex_v1 = __shfl_xor_sync(0xffffffff, v13, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k12 = ex_k0;
    k13 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v12 = ex_v0;
    v13 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k14;
    ex_k1 = __shfl_xor_sync(0xffffffff, k15, mask);
    ex_v0 = v14;
    ex_v1 = __shfl_xor_sync(0xffffffff, v15, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k14 = ex_k0;
    k15 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v14 = ex_v0;
    v15 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k16;
    ex_k1 = __shfl_xor_sync(0xffffffff, k17, mask);
    ex_v0 = v16;
    ex_v1 = __shfl_xor_sync(0xffffffff, v17, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k16 = ex_k0;
    k17 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v16 = ex_v0;
    v17 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k18;
    ex_k1 = __shfl_xor_sync(0xffffffff, k19, mask);
    ex_v0 = v18;
    ex_v1 = __shfl_xor_sync(0xffffffff, v19, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k18 = ex_k0;
    k19 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v18 = ex_v0;
    v19 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k20;
    ex_k1 = __shfl_xor_sync(0xffffffff, k21, mask);
    ex_v0 = v20;
    ex_v1 = __shfl_xor_sync(0xffffffff, v21, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k20 = ex_k0;
    k21 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v20 = ex_v0;
    v21 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k22;
    ex_k1 = __shfl_xor_sync(0xffffffff, k23, mask);
    ex_v0 = v22;
    ex_v1 = __shfl_xor_sync(0xffffffff, v23, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k22 = ex_k0;
    k23 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v22 = ex_v0;
    v23 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k24;
    ex_k1 = __shfl_xor_sync(0xffffffff, k25, mask);
    ex_v0 = v24;
    ex_v1 = __shfl_xor_sync(0xffffffff, v25, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k24 = ex_k0;
    k25 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v24 = ex_v0;
    v25 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k26;
    ex_k1 = __shfl_xor_sync(0xffffffff, k27, mask);
    ex_v0 = v26;
    ex_v1 = __shfl_xor_sync(0xffffffff, v27, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k26 = ex_k0;
    k27 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v26 = ex_v0;
    v27 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k28;
    ex_k1 = __shfl_xor_sync(0xffffffff, k29, mask);
    ex_v0 = v28;
    ex_v1 = __shfl_xor_sync(0xffffffff, v29, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k28 = ex_k0;
    k29 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v28 = ex_v0;
    v29 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    ex_k0 = k30;
    ex_k1 = __shfl_xor_sync(0xffffffff, k31, mask);
    ex_v0 = v30;
    ex_v1 = __shfl_xor_sync(0xffffffff, v31, mask);
    CMP_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if (bit)
        EQL_SWP(K, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k30 = ex_k0;
    k31 = __shfl_xor_sync(0xffffffff, ex_k1, mask);
    v30 = ex_v0;
    v31 = __shfl_xor_sync(0xffffffff, ex_v1, mask);
    if (bit)
        SWP(K, k0, k1, int, v0, v1);
    if (bit)
        SWP(K, k2, k3, int, v2, v3);
    if (bit)
        SWP(K, k4, k5, int, v4, v5);
    if (bit)
        SWP(K, k6, k7, int, v6, v7);
    if (bit)
        SWP(K, k8, k9, int, v8, v9);
    if (bit)
        SWP(K, k10, k11, int, v10, v11);
    if (bit)
        SWP(K, k12, k13, int, v12, v13);
    if (bit)
        SWP(K, k14, k15, int, v14, v15);
    if (bit)
        SWP(K, k16, k17, int, v16, v17);
    if (bit)
        SWP(K, k18, k19, int, v18, v19);
    if (bit)
        SWP(K, k20, k21, int, v20, v21);
    if (bit)
        SWP(K, k22, k23, int, v22, v23);
    if (bit)
        SWP(K, k24, k25, int, v24, v25);
    if (bit)
        SWP(K, k26, k27, int, v26, v27);
    if (bit)
        SWP(K, k28, k29, int, v28, v29);
    if (bit)
        SWP(K, k30, k31, int, v30, v31);
}

#endif
