//------------------------------------------------------------------------------
// GB_dense_ewise3_noaccum: C = A+B where A and B are dense, C is anything
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// C becomes a full matrix.

// FUTURE: extend to handle typecasting and generic operators.

#include "GB_dense.h"
#include "GB_binop.h"
#ifndef GBCOMPACT
#include "GB_binop__include.h"

#define GB_FREE_ALL ;

GrB_Info GB_dense_ewise3_noaccum    // C = A+B
(
    GrB_Matrix C,                   // input/output matrix
    const bool C_is_dense,          // true if C is dense on input
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_BinaryOp op,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C for dense C=A+B", GB0) ;
    ASSERT (!GB_PENDING (C)) ; ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_PENDING (A)) ; ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_PENDING (B)) ; ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (GB_IMPLIES (!C_is_dense, (C != A && C != B))) ;
    ASSERT (GB_is_dense (A)) ;
    ASSERT (GB_is_dense (B)) ;
    ASSERT_BINARYOP_OK (op, "op for dense C=A+B", GB0) ;
    ASSERT (op->ztype == C->type) ;
    ASSERT (op->xtype == A->type) ;
    ASSERT (op->ytype == B->type) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t anz = GB_NNZ (A) ;

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (2 * anz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // if C not already dense, allocate it as full
    //--------------------------------------------------------------------------

    // clear prior content and create C as a full matrix.  Keep the same type
    // and CSR/CSC for C.  Allocate the values of C but do not initialize them.

    if (!C_is_dense)
    { 
        // convert C to full; just allocate C->x.  Keep the dimensions of C.
        GB_OK (GB_to_full (C)) ;
    }
    else if (!GB_IS_FULL (C))
    {
        // C is dense, but not full; convert to full
        GB_sparse_to_full (C) ;
    }

    //--------------------------------------------------------------------------
    // define the worker for the switch factory
    //--------------------------------------------------------------------------

    #define GB_Cdense_ewise3_noaccum(op,xname) \
        GB_Cdense_ewise3_noaccum_ ## op ## xname

    #define GB_BINOP_WORKER(op,xname)                                       \
    {                                                                       \
        info = GB_Cdense_ewise3_noaccum(op,xname) (C, A, B, nthreads) ;     \
    }                                                                       \
    break ;

    //--------------------------------------------------------------------------
    // launch the switch factory
    //--------------------------------------------------------------------------

    GB_Opcode opcode ;
    GB_Type_code xcode, ycode, zcode ;
    if (GB_binop_builtin (A->type, false, B->type, false,
        op, false, &opcode, &xcode, &ycode, &zcode))
    { 
        #include "GB_binop_factory.c"
    }
    else
    {
        // this function is not called if the op cannot be applied
        ASSERT (0) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C=A+B output", GB0) ;
    return (GrB_SUCCESS) ;
}

#endif

