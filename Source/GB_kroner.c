//------------------------------------------------------------------------------
// GB_kroner: Kronecker product, C = kron (A,B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = kron(A,B) where op determines the binary multiplier to use.  The type of
// A and B are compatible with the x and y inputs of z=op(x,y), but can be
// different.  The type of C is the type of z.  C is hypersparse if either A
// or B are hypersparse.

// FUTURE: this would be faster with built-in types and operators.

// FUTURE: at most one thread is used for each vector of C=kron(A,B).  The
// matrix C is normally very large, but if both A and B are n-by-1, then C is
// n^2-by-1 and only a single thread is used.  A better method for this case
// would construct vectors of C in parallel.

// FUTURE: each vector C(:,k) takes O(nnz(C(:,k))) work, but this is not
// accounted for in the parallel load-balancing.

#define GB_FREE_WORKSPACE   \
{                           \
    GB_Matrix_free (&A2) ;  \
    GB_Matrix_free (&B2) ;  \
}

#define GB_FREE_ALL         \
{                           \
    GB_FREE_WORKSPACE ;     \
    GB_phybix_free (C) ;    \
}

#include "GB_kron.h"
#include "GB_emult.h"

GrB_Info GB_kroner                  // C = kron (A,B)
(
    GrB_Matrix C,                   // output matrix
    const bool C_is_csc,            // desired format of C
    const GrB_BinaryOp op,          // multiply operator
    const GrB_Matrix A_in,          // input matrix
    bool A_is_pattern,              // true if values of A are not used
    const GrB_Matrix B_in,          // input matrix
    bool B_is_pattern,              // true if values of B are not used
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;

    struct GB_Matrix_opaque A2_header, B2_header ;
    GrB_Matrix A2 = NULL, B2 = NULL ;

    ASSERT_MATRIX_OK (A_in, "A_in for kron (A,B)", GB0) ;
    ASSERT_MATRIX_OK (B_in, "B_in for kron (A,B)", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for kron (A,B)", GB0) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (A_in) ;
    GB_MATRIX_WAIT (B_in) ;

    //--------------------------------------------------------------------------
    // bitmap case: create sparse copies of A and B if they are bitmap
    //--------------------------------------------------------------------------

    GrB_Matrix A = A_in ;
    if (GB_IS_BITMAP (A))
    { 
        GBURBLE ("A:") ;
        // set A2->iso = A->iso     OK: no need for burble
        GB_CLEAR_STATIC_HEADER (A2, &A2_header) ;
        GB_OK (GB_dup_worker (&A2, A->iso, A, true, NULL, Context)) ;
        ASSERT_MATRIX_OK (A2, "dup A2 for kron (A,B)", GB0) ;
        GB_OK (GB_convert_bitmap_to_sparse (A2, Context)) ;
        ASSERT_MATRIX_OK (A2, "to sparse, A2 for kron (A,B)", GB0) ;
        A = A2 ;
    }

    GrB_Matrix B = B_in ;
    if (GB_IS_BITMAP (B))
    { 
        GBURBLE ("B:") ;
        // set B2->iso = B->iso     OK: no need for burble
        GB_CLEAR_STATIC_HEADER (B2, &B2_header) ;
        GB_OK (GB_dup_worker (&B2, B->iso, B, true, NULL, Context)) ;
        ASSERT_MATRIX_OK (B2, "dup B2 for kron (A,B)", GB0) ;
        GB_OK (GB_convert_bitmap_to_sparse (B2, Context)) ;
        ASSERT_MATRIX_OK (B2, "to sparse, B2 for kron (A,B)", GB0) ;
        B = B2 ;
    }

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    const GB_void *restrict Ax = A_is_pattern ? NULL : ((GB_void *) A->x) ;
    const int64_t asize = A->type->size ;
    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;
    int64_t anvec = A->nvec ;
    int64_t anz = GB_nnz (A) ;

    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int64_t *restrict Bi = B->i ;
    const GB_void *restrict Bx = B_is_pattern ? NULL : ((GB_void *) B->x) ;
    const int64_t bsize = B->type->size ;
    const int64_t bvlen = B->vlen ;
    const int64_t bvdim = B->vdim ;
    int64_t bnvec = B->nvec ;
    int64_t bnz = GB_nnz (B) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    double work = ((double) anz) * ((double) bnz)
                + (((double) anvec) * ((double) bnvec)) ;

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (work, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // check if C is iso and compute its iso value if it is
    //--------------------------------------------------------------------------

    GrB_Type ctype = op->ztype ;
    const size_t csize = ctype->size ;
    GB_void cscalar [GB_VLA(csize)] ;
    bool C_iso = GB_iso_emult (cscalar, ctype, A, B, op) ;

    //--------------------------------------------------------------------------
    // compute info of the output matrix C
    //--------------------------------------------------------------------------

    // C has the same type as z for the multiply operator, z=op(x,y)

    GrB_Index cvlen, cvdim, cnzmax, cnvec ;

    bool ok = GB_int64_multiply (&cvlen, avlen, bvlen) ;
    ok = ok & GB_int64_multiply (&cvdim, avdim, bvdim) ;
    ok = ok & GB_int64_multiply (&cnzmax, anz, bnz) ;
    ok = ok & GB_int64_multiply (&cnvec, anvec, bnvec) ;
    ASSERT (ok) ;

    if (C_iso)
    { 
        // the values of A and B are no longer needed if C is iso
        GBURBLE ("(iso kron) ") ;
        A_is_pattern = true ;
        B_is_pattern = true ;
    }

    // C is hypersparse if either A or B are hypersparse.  It is never bitmap.
    bool C_is_hyper = (cvdim > 1) && (Ah != NULL || Bh != NULL) ;

    //--------------------------------------------------------------------------
    // get operator
    //--------------------------------------------------------------------------

    GxB_binary_function fmult = op->binop_function ;
    GB_Opcode opcode = op->opcode ;
    bool op_is_positional = GB_OPCODE_IS_POSITIONAL (opcode) ;
    GB_cast_function cast_A = NULL, cast_B = NULL ;
    if (!A_is_pattern)
    { 
        cast_A = GB_cast_factory (op->xtype->code, A->type->code) ;
    }
    if (!B_is_pattern)
    { 
        cast_B = GB_cast_factory (op->ytype->code, B->type->code) ;
    }


    //--------------------------------------------------------------------------
    // count nonzero elements in result
    //--------------------------------------------------------------------------

    bool C_is_full = GB_as_if_full (A) && GB_as_if_full (B) ;
    const bool A_iso = A->iso ;
    const bool B_iso = B->iso ;

    GrB_Index cnz = 0 ;
    int64_t kC ;

    int64_t nvec_nonempty = 0 ;
    size_t p_size ;
    int64_t *restrict p = GB_MALLOC (cnvec + 1, int64_t, &(p_size)) ;
    ASSERT (p_size == GB_Global_memtable_size (p)) ;
    GB_memset (p, 0, p_size, nthreads) ;

    size_t h_size = 0, hp_size = 0 ;
    int64_t *restrict h = NULL ;
    int64_t *restrict hp = NULL ;


    if (!C_iso && !op_is_positional)
    {
        #pragma omp parallel for num_threads(nthreads) schedule(guided)
        for (kC = 0; kC < cnvec; kC++)
        {
            int64_t kA = kC / bnvec ;
            int64_t kB = kC % bnvec ;

            // get B(:,jB), the (kB)th vector of B
            const int64_t jB = GBH (Bh, kB) ;
            int64_t pB_start = GBP (Bp, kB, bvlen) ;
            int64_t pB_end = GBP (Bp, kB + 1, bvlen) ;
            int64_t bknz = pB_start - pB_end ;
            if (bknz == 0) continue ;
            GB_void bwork[GB_VLA(bsize)] ;
            if (!B_is_pattern && B_iso)
            {
                cast_B(bwork, Bx, bsize) ;
            }

            // get A(:,jA), the (kA)th vector of A
            const int64_t jA = GBH (Ah, kA) ;
            int64_t pA_start = GBP (Ap, kA, avlen) ;
            int64_t pA_end = GBP (Ap, kA + 1, avlen) ;
            GB_void awork[GB_VLA(asize)] ;
            if (!A_is_pattern && A_iso)
            {
                cast_A(awork, Ax, asize) ;
            }
            for (int64_t pA = pA_start; pA < pA_end; pA++)
            {
                // awork = A(iA,jA), typecasted to op->xtype
                int64_t iA = GBI (Ai, pA, avlen) ;
                int64_t iAblock = iA * bvlen ;
                if (!A_is_pattern && !A_iso)
                {
                    cast_A(awork, Ax + (pA * asize), asize);
                }
                for (int64_t pB = pB_start; pB < pB_end; pB++)
                {
                    // bwork = B(iB,jB), typecasted to op->ytype
                    int64_t iB = GBI (Bi, pB, bvlen) ;
                    if (!B_is_pattern && !B_iso)
                    {
                        cast_B(bwork, Bx + (pB * bsize), bsize) ;
                    }
                    // C(iC,jC) = A(iA,jA) * B(iB,jB)
                    // standard binary operator
                    GB_void cwork[GB_VLA(csize)] ;
                    fmult(cwork, awork, bwork) ;
                    for (size_t i = 0; i < csize; ++i)
                    {
                        if (*(cwork + i))
                        {
                            p [kC]++ ;
                            break;
                        }
                    }
                }
            }
        }

        GB_cumsum (p, cnvec, NULL, nthreads, Context);

        if (!(C_is_full = (C_is_full && cnz == cnzmax) ))
        { 
            if (C_is_hyper)
            { 
                h = GB_MALLOC (cnvec, int64_t, &(h_size)) ;
                hp = GB_MALLOC (cnvec, int64_t, &(hp_size)) ;
                ASSERT (h_size == GB_Global_memtable_size (h) && hp_size == GB_Global_memtable_size (hp)) ;
                GB_memset (h, 0, h_size, nthreads) ;
                GB_memset (hp, 0, hp_size, nthreads) ;
                for (kC = 0 ; kC < cnvec ; kC++)
                { 
                    if (p [kC + 1] > p [kC])
                    { 
                        int64_t kA = kC / bnvec ;
                        int64_t kB = kC % bnvec ;
                        const int64_t jA = GBH (Ah, kA) ;
                        const int64_t jB = GBH (Bh, kB) ;

                        h [nvec_nonempty++] = jA * bvdim + jB ;
                        hp [nvec_nonempty] = p [kC + 1] ;
                    }
                }
                cnz = hp[nvec_nonempty];
            }
            else
            {
                cnz = p[cnvec];
            }
        }

    }
    else if (!C_is_full)
    { 
        h = GB_MALLOC (cnvec, int64_t, &(h_size)) ;
        ASSERT (h_size == GB_Global_memtable_size (h)) ;
        #pragma omp parallel for num_threads(nthreads) schedule(guided)
        for (kC = 0 ; kC < cnvec ; kC++)
        {
            const int64_t kA = kC / bnvec ;
            const int64_t kB = kC % bnvec ;

            // get A(:,jA), the (kA)th vector of A
            const int64_t jA = GBH (Ah, kA) ;
            const int64_t aknz = (Ap == NULL) ? avlen : (Ap [kA+1] - Ap [kA]) ;
            // get B(:,jB), the (kB)th vector of B
            const int64_t jB = GBH (Bh, kB) ;
            const int64_t bknz = (Bp == NULL) ? bvlen : (Bp [kB+1] - Bp [kB]) ;
            // determine # entries in C(:,jC), the (kC)th vector of C
            // int64_t kC = kA * bnvec + kB ;

            p [kC] = aknz * bknz ;

            if (C_is_hyper)
            { 
                h [kC] = jA * bvdim + jB ;
            }
        }

        GB_cumsum (p, cnvec, &(C->nvec_nonempty), nthreads, Context) ;
        cnz = p[cnvec];
        if (C_is_hyper) nvec_nonempty = cnvec ;
    }

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    int sparsity = C_is_full ? GxB_FULL :
                   ((C_is_hyper) ? GxB_HYPERSPARSE : GxB_SPARSE) ;

    // TODO" C_is_hyper special case
    if (C_is_hyper)
    { 
        GB_OK (GB_new_bix(&C, // full, sparse, or hyper; existing header
                          ctype, (int64_t) cvlen, (int64_t) cvdim, GB_Ap_malloc, C_is_csc,
                          sparsity, true, B->hyper_switch, nvec_nonempty, cnz, true, C_iso, Context));
    }
    else
    { 
        GB_OK (GB_new_bix(&C, // full, sparse, or hyper; existing header
                          ctype, (int64_t) cvlen, (int64_t) cvdim, GB_Ap_malloc, C_is_csc,
                          sparsity, true, B->hyper_switch, cnvec, cnz, true, C_iso, Context));
    }

    //--------------------------------------------------------------------------
    // compute the column counts of C, and C->h if C is hypersparse
    //--------------------------------------------------------------------------

    int64_t *restrict Cp = C->p ;
    int64_t *restrict Ch = C->h ;
    int64_t *restrict Ci = C->i ;
    GB_void *restrict Cx = (GB_void *) C->x ;
    int64_t *restrict Cx_int64 = NULL ;
    int32_t *restrict Cx_int32 = NULL ;

    int64_t offset = 0 ;
    if (op_is_positional)
    { 
        offset = GB_positional_offset (opcode, NULL) ;
        Cx_int64 = (int64_t *) Cx ;
        Cx_int32 = (int32_t *) Cx ;
    }
    bool is64 = (ctype == GrB_INT64) ;

    if (!C_is_full)
    { 
        // if C_iso || op_is_pos || !C_is_hyper
        // Copy p
        // else
        // Copy hp
        if (C_is_hyper)
        { 
            C->nvec = nvec_nonempty ;
            C->nvec_nonempty = nvec_nonempty ;
            GB_memcpy (Ch, h, sizeof(int64_t) * nvec_nonempty, nthreads) ;
        }
        GB_FREE (&h, h_size) ;

        if (C_iso || op_is_positional || !C_is_hyper)
        { 
            GB_memcpy (Cp, p, p_size, nthreads) ;
            C->nvals = Cp [cnvec] ;
        }
        else
        { 
            GB_memcpy (Cp, hp, sizeof(int64_t) * (nvec_nonempty + 1), nthreads) ;
            C->nvals = Cp [nvec_nonempty] ;
            GB_FREE (&hp, hp_size) ;
        }
    }

    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // C = kron (A,B) where C is iso and full
    //--------------------------------------------------------------------------

    if (C_iso)
    {
        // Cx [0] = cscalar = op (A,B)
        memcpy (C->x, cscalar, csize) ;
        if (C_is_full)
        { 
            // no more work to do if C is iso and full
            ASSERT_MATRIX_OK (C, "C=kron(A,B), iso full", GB0) ;
            GB_FREE_WORKSPACE ;
            return (GrB_SUCCESS) ;
        }
    }

    //--------------------------------------------------------------------------
    // C = kron (A,B)
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(nthreads) schedule(guided)
    for (kC = 0 ; kC < cnvec ; kC++)
    {
        int64_t kA = kC / bnvec ;
        int64_t kB = kC % bnvec ;

        // get B(:,jB), the (kB)th vector of B
        int64_t jB = GBH (Bh, kB) ;
        int64_t pB_start = GBP (Bp, kB, bvlen) ;
        int64_t pB_end   = GBP (Bp, kB+1, bvlen) ;
        int64_t bknz = pB_start - pB_end ;
        if (bknz == 0) continue ;
        GB_void bwork [GB_VLA(bsize)] ;
        if (!B_is_pattern && B_iso)
        { 
            cast_B (bwork, Bx, bsize) ;
        }

        // get C(:,jC), the (kC)th vector of C
        int64_t pC = GBP (p, kC, cvlen) ;
        int64_t pC_next = GBP (p, kC + 1, cvlen) ;

        // get A(:,jA), the (kA)th vector of A
        int64_t jA = GBH (Ah, kA) ;
        int64_t pA_start = GBP (Ap, kA, avlen) ;
        int64_t pA_end   = GBP (Ap, kA+1, avlen) ;
        GB_void awork [GB_VLA(asize)] ;
        if (!A_is_pattern && A_iso)
        { 
            cast_A (awork, Ax, asize) ;
        }

        for (int64_t pA = pA_start ; pA < pA_end ; pA++)
        {
            // awork = A(iA,jA), typecasted to op->xtype
            int64_t iA = GBI (Ai, pA, avlen) ;
            int64_t iAblock = iA * bvlen ;
            if (!A_is_pattern && !A_iso)
            { 
                cast_A (awork, Ax + (pA*asize), asize) ;
            }
            for (int64_t pB = pB_start ; pB < pB_end && pC < pC_next ; pB++)
            {
                // bwork = B(iB,jB), typecasted to op->ytype
                int64_t iB = GBI (Bi, pB, bvlen) ;
                if (!B_is_pattern && !B_iso)
                { 
                    cast_B (bwork, Bx +(pB*bsize), bsize) ;
                }
                // C(iC,jC) = A(iA,jA) * B(iB,jB)
                if (!C_is_full)
                { 
                    int64_t iC = iAblock + iB ;
                    Ci [pC] = iC ;
                }
                if (op_is_positional)
                {
                    // positional binary operator
                    switch (opcode)
                    {
                        case GB_FIRSTI_binop_code   : 
                            // z = first_i(A(iA,jA),y) == iA
                        case GB_FIRSTI1_binop_code  : 
                            // z = first_i1(A(iA,jA),y) == iA+1
                            if (is64)
                            { 
                                Cx_int64 [pC] = iA + offset ;
                            }
                            else
                            { 
                                Cx_int32 [pC] = (int32_t) (iA + offset) ;
                            }
                            break ;
                        case GB_FIRSTJ_binop_code   : 
                            // z = first_j(A(iA,jA),y) == jA
                        case GB_FIRSTJ1_binop_code  : 
                            // z = first_j1(A(iA,jA),y) == jA+1
                            if (is64)
                            { 
                                Cx_int64 [pC] = jA + offset ;
                            }
                            else
                            { 
                                Cx_int32 [pC] = (int32_t) (jA + offset) ;
                            }
                            break ;
                        case GB_SECONDI_binop_code  : 
                            // z = second_i(x,B(iB,jB)) == iB
                        case GB_SECONDI1_binop_code : 
                            // z = second_i1(x,B(iB,jB)) == iB+1
                            if (is64)
                            { 
                                Cx_int64 [pC] = iB + offset ;
                            }
                            else
                            { 
                                Cx_int32 [pC] = (int32_t) (iB + offset) ;
                            }
                            break ;
                        case GB_SECONDJ_binop_code  : 
                            // z = second_j(x,B(iB,jB)) == jB
                        case GB_SECONDJ1_binop_code : 
                            // z = second_j1(x,B(iB,jB)) == jB+1
                            if (is64)
                            { 
                                Cx_int64 [pC] = jB + offset ;
                            }
                            else
                            { 
                                Cx_int32 [pC] = (int32_t) (jB + offset) ;
                            }
                            break ;
                        default: ;
                    }
                    pC++ ;
                }
                else if (!C_iso)
                { 
                    // standard binary operator
                    fmult (Cx +(pC*csize), awork, bwork) ;
                    for (size_t i = 0 ; i < csize ; ++i)
                    { 
                        if (*(Cx + (pC*csize + i)))
                        { 
                            pC++ ;
                            break;
                        }
                    }
                } else
                { 
                    pC++ ;
                }

            }
        }
    }
    GB_FREE (&p, p_size) ;

    //--------------------------------------------------------------------------
    // remove empty vectors from C, if hypersparse
    //--------------------------------------------------------------------------

    GB_OK (GB_hypermatrix_prune (C, Context)) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C=kron(A,B)", GB0) ;
    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

