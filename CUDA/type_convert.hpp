/*
 * Copyright (c) 2019,2020 NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GB_CONV_TYPE_H
#define GB_CONV_TYPE_H

extern "C" {
#include "GB.h"
};
#include <stdint.h>

/**---------------------------------------------------------------------------*
 * @file type_convert.hpp
 * @brief Defines the mapping between concrete C++ types and Grb types.
 *---------------------------------------------------------------------------**/
namespace cuda {

template <typename T>
GrB_Type to_grb_type();

template<> GrB_Type to_grb_type<int8_t>() { return GrB_INT8; }
template<> GrB_Type to_grb_type<int16_t>() { return GrB_INT16; }
template<> GrB_Type to_grb_type<int32_t>() { return GrB_INT32; }
template<> GrB_Type to_grb_type<int64_t>() { return GrB_INT64; }
template<> GrB_Type to_grb_type<uint8_t>() { return GrB_UINT8; }
template<> GrB_Type to_grb_type<uint16_t>() { return GrB_UINT16; }
template<> GrB_Type to_grb_type<uint32_t>() { return GrB_UINT32; }
template<> GrB_Type to_grb_type<uint64_t>() { return GrB_UINT64; }
template<> GrB_Type to_grb_type<float>() { return GrB_FP32; }
template<> GrB_Type to_grb_type<double>() { return GrB_FP64; }
template<> GrB_Type to_grb_type<bool>() { return GrB_BOOL; }


template <typename T>
void set_element(GrB_Matrix A, T x, int64_t i, int64_t j);

template<> void set_element<int8_t>(GrB_Matrix A, int8_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_INT8(A, x, i, j); }
template<> void set_element<int16_t>(GrB_Matrix A, int16_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_INT16(A, x, i, j); }
template<> void set_element<int32_t>(GrB_Matrix A, int32_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_INT32(A, x, i, j); }
template<> void set_element<int64_t>(GrB_Matrix A, int64_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_INT64(A, x, i, j); }
template<> void set_element<uint8_t>(GrB_Matrix A, uint8_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_UINT8(A, x, i, j); }
template<> void set_element<uint16_t>(GrB_Matrix A, uint16_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_UINT16(A, x, i, j); }
template<> void set_element<uint32_t>(GrB_Matrix A, uint32_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_UINT32(A, x, i, j); }
template<> void set_element<uint64_t>(GrB_Matrix A, uint64_t x, int64_t i, int64_t j) { GrB_Matrix_setElement_UINT64(A, x, i, j); }
template<> void set_element<float>(GrB_Matrix A, float x, int64_t i, int64_t j) { GrB_Matrix_setElement_FP32(A, x, i, j); }
template<> void set_element<double>(GrB_Matrix A, double x, int64_t i, int64_t j) { GrB_Matrix_setElement_FP64(A, x, i, j); }
template<> void set_element<bool>(GrB_Matrix A, bool x, int64_t i, int64_t j) { GrB_Matrix_setElement_BOOL(A, x, i, j); }


template <typename T>
void vector_set_element(GrB_Vector A, T x, int64_t i);

template<> void vector_set_element<int8_t>(GrB_Vector A, int8_t x, int64_t i) { GrB_Vector_setElement_INT8(A, x, i); }
template<> void vector_set_element<int16_t>(GrB_Vector A, int16_t x, int64_t i) { GrB_Vector_setElement_INT16(A, x, i); }
template<> void vector_set_element<int32_t>(GrB_Vector A, int32_t x, int64_t i) { GrB_Vector_setElement_INT32(A, x, i); }
template<> void vector_set_element<int64_t>(GrB_Vector A, int64_t x, int64_t i) { GrB_Vector_setElement_INT64(A, x, i); }
template<> void vector_set_element<uint8_t>(GrB_Vector A, uint8_t x, int64_t i) { GrB_Vector_setElement_UINT8(A, x, i); }
template<> void vector_set_element<uint16_t>(GrB_Vector A, uint16_t x, int64_t i) { GrB_Vector_setElement_UINT16(A, x, i); }
template<> void vector_set_element<uint32_t>(GrB_Vector A, uint32_t x, int64_t i) { GrB_Vector_setElement_UINT32(A, x, i); }
template<> void vector_set_element<uint64_t>(GrB_Vector A, uint64_t x, int64_t i) { GrB_Vector_setElement_UINT64(A, x, i); }
template<> void vector_set_element<float>(GrB_Vector A, float x, int64_t i) { GrB_Vector_setElement_FP32(A, x, i); }
template<> void vector_set_element<double>(GrB_Vector A, double x, int64_t i) { GrB_Vector_setElement_FP64(A, x, i); }
template<> void vector_set_element<bool>(GrB_Vector A, bool x, int64_t i) { GrB_Vector_setElement_BOOL(A, x, i); }


template<typename T>
void vector_reduce(T *scalar, GrB_Vector A, GrB_Monoid op);

template<> void vector_reduce<int8_t>(int8_t *scalar, GrB_Vector A, GrB_Monoid op) { GrB_Vector_reduce_INT8(scalar, NULL, op, A, NULL); }
template<> void vector_reduce<int16_t>(int16_t *scalar, GrB_Vector A, GrB_Monoid op) { GrB_Vector_reduce_INT16(scalar, NULL, op, A, NULL); }
template<> void vector_reduce<int32_t>(int32_t *scalar, GrB_Vector A, GrB_Monoid op) { GrB_Vector_reduce_INT32(scalar, NULL, op, A, NULL); }
template<> void vector_reduce<int64_t>(int64_t *scalar, GrB_Vector A, GrB_Monoid op) { GrB_Vector_reduce_INT64(scalar, NULL, op, A, NULL); }
template<> void vector_reduce<uint8_t>(uint8_t *scalar, GrB_Vector A, GrB_Monoid op) { GrB_Vector_reduce_UINT8(scalar, NULL, op, A, NULL); }
template<> void vector_reduce<uint16_t>(uint16_t *scalar, GrB_Vector A, GrB_Monoid op) { GrB_Vector_reduce_UINT16(scalar, NULL, op, A, NULL); }
template<> void vector_reduce<uint32_t>(uint32_t *scalar, GrB_Vector A, GrB_Monoid op) { GrB_Vector_reduce_UINT32(scalar, NULL, op, A, NULL); }
template<> void vector_reduce<uint64_t>(uint64_t *scalar, GrB_Vector A, GrB_Monoid op) { GrB_Vector_reduce_UINT64(scalar, NULL, op, A, NULL); }
template<> void vector_reduce<float>(float *scalar, GrB_Vector A, GrB_Monoid op) { GrB_Vector_reduce_FP32(scalar, NULL, op, A, NULL); }
template<> void vector_reduce<double>(double *scalar, GrB_Vector A, GrB_Monoid op) { GrB_Vector_reduce_FP64(scalar, NULL, op, A, NULL); }
template<> void vector_reduce<bool>(bool *scalar, GrB_Vector A, GrB_Monoid op) { GrB_Vector_reduce_BOOL(scalar, NULL, op, A, NULL); }

template <typename T>
GrB_Info get_element(GrB_Matrix A, T* x, int64_t i, int64_t j);
template<> GrB_Info get_element<int8_t>(GrB_Matrix A, int8_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_INT8(x, A, i, j); }
template<> GrB_Info get_element<int16_t>(GrB_Matrix A, int16_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_INT16(x, A, i, j); }
template<> GrB_Info get_element<int32_t>(GrB_Matrix A, int32_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_INT32(x, A, i, j); }
template<> GrB_Info get_element<int64_t>(GrB_Matrix A, int64_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_INT64(x, A, i, j); }
template<> GrB_Info get_element<uint8_t>(GrB_Matrix A, uint8_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_UINT8(x, A, i, j); }
template<> GrB_Info get_element<uint16_t>(GrB_Matrix A, uint16_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_UINT16(x, A, i, j); }
template<> GrB_Info get_element<uint32_t>(GrB_Matrix A, uint32_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_UINT32(x, A, i, j); }
template<> GrB_Info get_element<uint64_t>(GrB_Matrix A, uint64_t *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_UINT64(x, A, i, j); }
template<> GrB_Info get_element<float>(GrB_Matrix A, float *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_FP32(x, A, i, j); }
template<> GrB_Info get_element<double>(GrB_Matrix A, double *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_FP64(x, A, i, j); }
template<> GrB_Info get_element<bool>(GrB_Matrix A, bool *x, int64_t i, int64_t j) { return GrB_Matrix_extractElement_BOOL(x, A, i, j); }


}  // namespace cuda
#endif
