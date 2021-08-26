//------------------------------------------------------------------------------
// GB_serialize.h: definitions for GB_serialize_* and deserialize methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SERIALIZE_H
#define GB_SERIALIZE_H

GrB_Info GB_serialize               // serialize a matrix into a blob
(
    // output:
    GB_void **blob_handle,          // serialized matrix, allocated on output
    size_t *blob_size_handle,       // size of the blob
    // input:
    const GrB_Matrix A,             // matrix to serialize
    int32_t method,                 // method to use
    GB_Context Context
) ;

GrB_Info GB_deserialize             // deserialize a matrix from a blob
(
    // output:
    GrB_Matrix *Chandle,            // output matrix created from the blob
    // input:
    const GB_void *blob,            // serialized matrix 
    size_t blob_size,               // size of the blob
    GrB_Type user_type,             // type of matrix, if user-defined
    GB_Context Context
) ;

typedef struct
{
    // 7x8 = 56 bytes
    size_t blob_size ;              // size of the blob
    int64_t vlen ;                  // length of each vector
    int64_t vdim ;                  // # of vectors
    int64_t nvec ;                  // size of Ap is nvec+1, Ah is nvec
    int64_t nvec_nonempty ;         // # of non-empty vectors
    int64_t nvals ;                 // # of values if bitmap
    size_t typesize ;               // size of the type

    // 5x4 = 20 bytes
    int32_t version ;               // SuiteSparse:GraphBLAS version
    GB_Type_code typecode ;         // matrix type code
    float hyper_switch ;            // hyper-switch control
    float bitmap_switch ;           // bitmap-switch control
    int32_t sparsity_control ;      // sparsity control

    // 8 bytes
    int32_t sparsity ;              // hyper/sparse/bitmap/full
    bool iso ;                      // true if matrix is iso
    bool is_csc ;                   // true if by column, false if by row
    bool unused1, unused2 ;
}
GB_blob_header ;

typedef struct
{
    void *p ;                       // pointer to the compressed block
    size_t p_size ;                 // size of compressed block, or zero
                                    // if p is not malloc'ed
    // after the blocks are compressed, these 2 terms are overwritten with
    // their cumulative sum:
    size_t uncompressed ;           // original size of the block
    size_t compressed ;             // size of the block when compressed
}
GB_blocks ;

GrB_Info GB_serialize_array
(
    // output:
    GB_blocks **Blocks_handle,          // Blocks: array of size nblocks+1
    size_t *Blocks_size_handle,         // size of Blocks
    int32_t *nblocks_handle,            // # of blocks
    int32_t *method_used,               // method used
    // input:
    GB_void *X,                         // input array of size len
    size_t len,                         // size of X, in bytes
    int32_t method,                     // compression method requested
    GB_Context Context
) ;

void GB_serialize_free_blocks
(
    GB_blocks **Blocks_handle,      // array of size nblocks
    size_t Blocks_size,             // size of Blocks
    int32_t nblocks,                // # of blocks, or zero if no blocks
    GB_Context Context
) ;

void GB_serialize_to_blob
(
    // input/output
    GB_void *blob,          // blocks are appended to the blob
    size_t *s_handle,       // location to append into the blob
    // input:
    GB_blocks *Blocks,      // Blocks: array of size nblocks
    int32_t nblocks,        // # of blocks
    int32_t method_used,    // compression method used
    int nthreads_max        // # of threads to use
) ;

GrB_Info GB_deserialize_from_blob
(
    // output:
    GB_void **X_handle,         // uncompressed output array
    size_t *X_size_handle,      // size of X
    // input:
    const GB_void *blob,
    size_t blob_size,
    // input/output:
    size_t *s_handle,           // location to write into the blob
    GB_Context Context
) ;

#endif

