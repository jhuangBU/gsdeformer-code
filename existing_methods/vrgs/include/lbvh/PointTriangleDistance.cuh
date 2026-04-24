#ifndef LBVH_POINT_TRIANGLE_DISTANCE_CUH
#define LBVH_POINT_TRIANGLE_DISTANCE_CUH

#include "MathTools.h"

namespace lbvh {

template <class T>
inline __host__ __device__ void solveLdlt2D(
    const T A00, const T A01, const T A11, const T b0, const T b1, T& x0, T& x1) {
    T D00, D11, L01;
    D00 = A00;
    L01 = A01 / D00;
    D11 = A11 - L01 * L01 * D00;

    T y0, y1;
    y0 = b0;
    y1 = b1 - L01 * y0;

    x1 = y1 / D11;
    x0 = (y0 - D00 * L01 * x1) / D00;
}

template <class T>
inline __host__ __device__ int point_triangle_distance_type(
    const T* p, const T* t0, const T* t1, const T* t2) {
    T basis_row0[3], basis_row1[3], nVec[3];
    basis_row0[0] = t1[0] - t0[0];
    basis_row0[1] = t1[1] - t0[1];
    basis_row0[2] = t1[2] - t0[2];
    basis_row1[0] = t2[0] - t0[0];
    basis_row1[1] = t2[1] - t0[1];
    basis_row1[2] = t2[2] - t0[2];
    cross_product3(basis_row0, basis_row1, nVec);
    cross_product3(basis_row0, nVec, basis_row1);

    T param_col0[2], param_col1[2], param_col2[2];
    T sys[4], rhs[2], b[3];
    sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1]
        + basis_row0[2] * basis_row0[2];
    sys[1] = sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1]
        + basis_row0[2] * basis_row1[2];
    sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1]
        + basis_row1[2] * basis_row1[2];

    b[0] = p[0] - t0[0];
    b[1] = p[1] - t0[1];
    b[2] = p[2] - t0[2];

    rhs[0] = dot<3>(basis_row0, b);
    rhs[1] = dot<3>(basis_row1, b);
    solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1], param_col0[0], param_col0[1]);
    if (param_col0[0] > 0.0 && param_col0[0] < 1.0 && param_col0[1] >= 0.0) {
        return 3; // PE t0t1
    } else {
        basis_row0[0] = t2[0] - t1[0];
        basis_row0[1] = t2[1] - t1[1];
        basis_row0[2] = t2[2] - t1[2];
        cross_product3(basis_row0, nVec, basis_row1);

        sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1]
            + basis_row0[2] * basis_row0[2];
        sys[1] = sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1]
            + basis_row0[2] * basis_row1[2];
        sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1]
            + basis_row1[2] * basis_row1[2];
        b[0] = p[0] - t1[0];
        b[1] = p[1] - t1[1];
        b[2] = p[2] - t1[2];
        rhs[0] = dot<3>(basis_row0, b);
        rhs[1] = dot<3>(basis_row1, b);
        solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1], param_col1[0], param_col1[1]);
        if (param_col1[0] > 0.0 && param_col1[0] < 1.0 && param_col1[1] >= 0.0) {
            return 4; // PE t1t2
        } else {
            basis_row0[0] = t0[0] - t2[0];
            basis_row0[1] = t0[1] - t2[1];
            basis_row0[2] = t0[2] - t2[2];
            cross_product3(basis_row0, nVec, basis_row1);
            sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1]
                + basis_row0[2] * basis_row0[2];
            sys[1] = sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1]
                + basis_row0[2] * basis_row1[2];
            sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1]
                + basis_row1[2] * basis_row1[2];
            b[0] = p[0] - t2[0];
            b[1] = p[1] - t2[1];
            b[2] = p[2] - t2[2];
            rhs[0] = dot<3>(basis_row0, b);
            rhs[1] = dot<3>(basis_row1, b);
            solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1], param_col2[0], param_col2[1]);

            if (param_col2[0] > 0.0 && param_col2[0] < 1.0 && param_col2[1] >= 0.0) {
                return 5; // PE t2t0
            } else {
                if (param_col0[0] <= 0.0 && param_col2[0] >= 1.0) {
                    return 0; // PP t0
                } else if (param_col1[0] <= 0.0 && param_col0[0] >= 1.0) {
                    return 1; // PP t1
                } else if (param_col2[0] <= 0.0 && param_col1[0] >= 1.0) {
                    return 2; // PP t2
                } else {
                    return 6; // PT
                }
            }
        }
    }
}

template <class T>
inline __host__ __device__ T point_point_distance(const T* a, const T* b, const int dim = 3) {
    T dist = 0;
    for (int i = 0; i < dim; i++)
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    return dist;
}

template <class T>
inline __host__ __device__ T point_line_distance(const T* p, const T* e0, const T* e1, const int dim = 3) {
    if (dim == 2) {
        T e0e1[2];
        e0e1[0] = e1[0] - e0[0];
        e0e1[1] = e1[1] - e0[1];
        T numerator = (e0e1[1] * p[0] - e0e1[0] * p[1] + e1[0] * e0[1] - e1[1] * e0[0]);
        return numerator * numerator / norm_sqr<2>(e0e1);
    } else {
        // return (e0 - p).cross(e1 - p).squaredNorm() / (e1 - e0).squaredNorm();
        T pe0[3], pe1[3], e0e1[3];
        for (int i = 0; i < dim; i++) {
            pe0[i] = e0[i] - p[i];
            pe1[i] = e1[i] - p[i];
            e0e1[i] = e1[i] - e0[i];
        }
        T nor[3];
        cross_product3(pe0, pe1, nor);
        return norm_sqr<3>(nor) / norm_sqr<3>(e0e1);
    }
}

template <class T>
inline __host__ __device__ T point_plane_distance(const T* p, const T* t0, const T* t1, const T* t2) {
    // const Eigen::Matrix<T, 3, 1> b = (t1 - t0).cross(t2 - t0);
    // T aTb = (p - t0).dot(b);
    // return aTb * aTb / b.squaredNorm();

    T t0t1[3], t0t2[3], t0p[3], b[3];
    for (int i = 0; i < 3; i++) {
        t0t1[i] = t1[i] - t0[i];
        t0t2[i] = t2[i] - t0[i];
        t0p[i] = p[i] - t0[i];
    }
    cross_product3(t0t1, t0t2, b);
    T aTb = dot<3>(t0p, b);
    return aTb * aTb / norm_sqr<3>(b);
}

template <class T>
inline __host__ __device__ T point_triangle_distance(const T* p, const T* t0, const T* t1, const T* t2) {
    switch (point_triangle_distance_type(p, t0, t1, t2)) {
        case 0:
            return point_point_distance(p, t0);
        case 1:
            return point_point_distance(p, t1);
        case 2:
            return point_point_distance(p, t2);
        case 3:
            return point_line_distance(p, t0, t1);
        case 4:
            return point_line_distance(p, t1, t2);
        case 5:
            return point_line_distance(p, t2, t0);
        case 6:
            return point_plane_distance(p, t0, t1, t2);
        default:
            return 1e20;
    }
}

}

#endif