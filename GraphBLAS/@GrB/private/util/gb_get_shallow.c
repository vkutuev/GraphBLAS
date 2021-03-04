//------------------------------------------------------------------------------
// gb_get_shallow: create a shallow copy of a MATLAB sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A = gb_get_shallow (X) constructs a shallow GrB_Matrix from a MATLAB
// mxArray, which can either be a MATLAB sparse matrix (double, complex, or
// logical) or a MATLAB struct that contains a GraphBLAS matrix.

// X must not be NULL, but it can be an empty matrix, as X = [ ] or even X = ''
// (the empty string).  In this case, A is returned as NULL.  This is not an
// error here, since the caller might be getting an optional input matrix, such
// as Cin or the Mask.

// FUTURE: it would be better to use the GxB* import/export functions,
// instead of accessing the opaque content of the GrB_Matrix directly.

#include "gb_matlab.h"

#define IF(error,message) \
    CHECK_ERROR (error, "invalid GraphBLAS struct (" message ")" ) ;

GrB_Matrix gb_get_shallow   // return a shallow copy of MATLAB sparse matrix
(
    const mxArray *X
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (X == NULL, "matrix missing") ;

    //--------------------------------------------------------------------------
    // construct the shallow GrB_Matrix
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL ;

    if (gb_mxarray_is_empty (X))
    { 

        //----------------------------------------------------------------------
        // matrix is empty
        //----------------------------------------------------------------------

        // X is a 0-by-0 MATLAB matrix.  Create a new 0-by-0 matrix of the same
        // type as X, with the default format.
        OK (GrB_Matrix_new (&A, gb_mxarray_type (X), 0, 0)) ;

    }
    else if (mxIsStruct (X))
    { 

        //----------------------------------------------------------------------
        // construct a shallow GrB_Matrix copy from a MATLAB struct
        //----------------------------------------------------------------------

        bool GraphBLASv3 = false ;
        mxArray *mx_type = NULL ;

        // get the type
        mx_type = mxGetField (X, 0, "GraphBLASv4") ;
        if (mx_type == NULL)
        {
            // check if it is a GraphBLASv3 struct
            mx_type = mxGetField (X, 0, "GraphBLAS") ;
            CHECK_ERROR (mx_type == NULL, "not a GraphBLAS struct") ;
            GraphBLASv3 = true ;
        }

        GrB_Type type = gb_mxstring_to_type (mx_type) ;
        size_t type_size ;
        OK (GxB_Type_size (&type_size, type)) ;

        // get the scalar info
        mxArray *opaque = mxGetField (X, 0, "s") ;
        IF (opaque == NULL, ".s missing") ;
        IF (mxGetM (opaque) != 1, ".s wrong size") ;
        IF (mxGetN (opaque) != (GraphBLASv3 ? 8 : 9), ".s wrong size") ;
        int64_t *s = mxGetInt64s (opaque) ;
        int64_t plen          = s [0] ;
        int64_t vlen          = s [1] ;
        int64_t vdim          = s [2] ;
        int64_t nvec          = s [3] ;
        int64_t nvec_nonempty = s [4] ;
        bool    by_col        = (bool) (s [6]) ;
        int64_t nzmax         = s [7] ;

        int sparsity_status, sparsity_control ;
        int64_t nvals ;
        if (GraphBLASv3)
        {
            // GraphBLASv3 struct: sparse or hypersparse only
            sparsity_control = GxB_AUTO_SPARSITY ;
            nvals            = 0 ;
        }
        else
        {
            // GraphBLASv4 struct: sparse, hypersparse, bitmap, or full
            sparsity_control = (int) (s [5]) ;
            nvals            = s [8] ;
        }

        int nfields = mxGetNumberOfFields (X) ;
        switch (nfields)
        {
            case 3 :
                // A is full, with 3 fields: GraphBLASv4, s, x
                sparsity_status = GxB_FULL ;
                break ;

            case 5 :
                // A is sparse, with 5 fields: GraphBLASv4, s, x, p, i
                sparsity_status = GxB_SPARSE ;
                break ;

            case 6 :
                // A is hypersparse, with 6 fields: GraphBLASv4, s, x, p, i, h
                sparsity_status = GxB_HYPERSPARSE ;
                break ;

            case 4 :
                // A is bitmap, with 4 fields: GraphBLASv4, s, x, b
                sparsity_status = GxB_BITMAP ;
                break ;

            default : ERROR ("invalid GraphBLAS struct") ;
        }

        // each component
        int64_t *Ap = NULL, *Ai = NULL, *Ah = NULL ;
        int8_t *Ab = NULL ;
        void *Ax = NULL ;

        // size of each component
        size_t Ap_size = 0 ;
        size_t Ah_size = 0 ;
        size_t Ab_size = 0 ;
        size_t Ai_size = 0 ;
        size_t Ax_size = 0 ;

        if (sparsity_status == GxB_HYPERSPARSE || sparsity_status == GxB_SPARSE)
        {
            // A is hypersparse or sparse

            // get Ap
            mxArray *Ap_mx = mxGetField (X, 0, "p") ;
            IF (Ap_mx == NULL, ".p missing") ;
            IF (mxGetM (Ap_mx) != 1, ".p wrong size") ;
            Ap = mxGetInt64s (Ap_mx) ;
            IF (Ap == NULL, ".p wrong type") ;
            Ap_size = mxGetN (Ap_mx) ;

            // get Ai
            mxArray *Ai_mx = mxGetField (X, 0, "i") ;
            IF (Ai_mx == NULL, ".i missing") ;
            IF (mxGetM (Ai_mx) != 1, ".i wrong size") ;
            Ai_size = mxGetN (Ai_mx) ;
            Ai = (Ai_size == 0) ? NULL : mxGetInt64s (Ai_mx) ;
            IF (Ai == NULL && Ai_size > 0, ".i wrong type") ;
        }

        // get the values
        mxArray *Ax_mx = mxGetField (X, 0, "x") ;
        IF (Ax_mx == NULL, ".x missing") ;
        IF (mxGetM (Ax_mx) != 1, ".x wrong size") ;
        Ax_size = mxGetN (Ax_mx) / type_size ;
        Ax = (Ax_size == 0) ? NULL : ((void *) mxGetUint8s (Ax_mx)) ;
        IF (Ax == NULL && Ax_size > 0, ".x wrong type") ;

        if (sparsity_status == GxB_HYPERSPARSE)
        { 
            // A is hypersparse
            // get the hyperlist
            mxArray *Ah_mx = mxGetField (X, 0, "h") ;
            IF (Ah_mx == NULL, ".h missing") ;
            IF (mxGetM (Ah_mx) != 1, ".h wrong size") ;
            Ah_size = mxGetN (Ah_mx) ;
            Ah = (Ah_size == 0) ? NULL : ((int64_t *) mxGetInt64s (Ah_mx)) ;
            IF (Ah == NULL && Ah_size > 0, ".h wrong type") ;
        }

        if (sparsity_status == GxB_BITMAP)
        { 
            // A is bitmap
            // get the bitmap
            mxArray *Ab_mx = mxGetField (X, 0, "b") ;
            IF (Ab_mx == NULL, ".b missing") ;
            IF (mxGetM (Ab_mx) != 1, ".b wrong size") ;
            Ab_size = mxGetN (Ab_mx) ;
            Ab = (Ab_size == 0) ? NULL : ((int8_t *) mxGetInt8s (Ab_mx)) ;
            IF (Ab == NULL && Ab_size > 0, ".b wrong type") ;
        }

        //----------------------------------------------------------------------
        // import the matrix
        //----------------------------------------------------------------------

        int64_t nrows = (by_col) ? vlen : vdim ;
        int64_t ncols = (by_col) ? vdim : vlen ;

        switch (sparsity_status)
        {
            case GxB_FULL :
                if (by_col)
                {
                    OK (GxB_Matrix_import_FullC (&A, type, nrows, ncols,
                        &Ax, Ax_size, NULL)) ;
                }
                else
                {
                    OK (GxB_Matrix_import_FullR (&A, type, nrows, ncols,
                        &Ax, Ax_size, NULL)) ;
                }
                break ;

            case GxB_SPARSE :
                if (by_col)
                {
                    OK (GxB_Matrix_import_CSC (&A, type, nrows, ncols,
                        &Ap, &Ai, &Ax, Ap_size, Ai_size, Ax_size,
                        false, NULL)) ;
                }
                else
                {
                    OK (GxB_Matrix_import_CSR (&A, type, nrows, ncols,
                        &Ap, &Ai, &Ax, Ap_size, Ai_size, Ax_size,
                        false, NULL)) ;
                }
                break ;

            case GxB_HYPERSPARSE :
                if (by_col)
                {
                    OK (GxB_Matrix_import_HyperCSC (&A, type, nrows, ncols,
                        &Ap, &Ah, &Ai, &Ax, Ap_size, Ah_size, Ai_size, Ax_size,
                        nvec, false, NULL)) ;
                }
                else
                {
                    OK (GxB_Matrix_import_HyperCSR (&A, type, nrows, ncols,
                        &Ap, &Ah, &Ai, &Ax, Ap_size, Ah_size, Ai_size, Ax_size,
                        nvec, false, NULL)) ;
                }
                break ;

            case GxB_BITMAP :
                if (by_col)
                {
                    OK (GxB_Matrix_import_BitmapC (&A, type, nrows, ncols,
                        &Ab, &Ax, Ab_size, Ax_size, nvals, NULL)) ;
                }
                else
                {
                    OK (GxB_Matrix_import_BitmapR (&A, type, nrows, ncols,
                        &Ab, &Ax, Ab_size, Ax_size, nvals, NULL)) ;
                }
                break ;

            default: ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // construct a shallow GrB_Matrix copy of a MATLAB matrix
        //----------------------------------------------------------------------

        // get the type and dimensions
        bool X_is_sparse = mxIsSparse (X) ;

        GrB_Type type = gb_mxarray_type (X) ;
        GrB_Index nrows = (GrB_Index) mxGetM (X) ;
        GrB_Index ncols = (GrB_Index) mxGetN (X) ;

        // get Xp, Xi, nzmax, or create them
        GrB_Index *Xp, *Xi, nzmax ;
        if (X_is_sparse)
        { 
            // get the nzmax, Xp, and Xi from the MATLAB sparse matrix X
            nzmax = (GrB_Index) mxGetNzmax (X) ;
            Xp = (GrB_Index *) mxGetJc (X) ;
            Xi = (GrB_Index *) mxGetIr (X) ;
        }
        else
        { 
            // X is a MATLAB full matrix; so is the GrB_Matrix
            nzmax = nrows * ncols ;
            Xp = NULL ;
            Xi = NULL ;
        }

        // get the numeric data
        void *Xx = NULL ;
        if (type == GrB_FP64)
        { 
            // MATLAB sparse or full double matrix
            Xx = mxGetDoubles (X) ;
        }
        else if (type == GxB_FC64)
        { 
            // MATLAB sparse or full double complex matrix
            Xx = mxGetComplexDoubles (X) ;
        }
        else if (type == GrB_BOOL)
        { 
            // MATLAB sparse or full logical matrix
            Xx = mxGetData (X) ;
        }
        else if (X_is_sparse)
        {
            // MATLAB does not support any other kinds of sparse matrices
            ERROR ("unsupported type") ;
        }
        else if (type == GrB_INT8)
        { 
            // full int8 matrix
            Xx = mxGetInt8s (X) ;
        }
        else if (type == GrB_INT16)
        { 
            // full int16 matrix
            Xx = mxGetInt16s (X) ;
        }
        else if (type == GrB_INT32)
        { 
            // full int32 matrix
            Xx = mxGetInt32s (X) ;
        }
        else if (type == GrB_INT64)
        { 
            // full int64 matrix
            Xx = mxGetInt64s (X) ;
        }
        else if (type == GrB_UINT8)
        { 
            // full uint8 matrix
            Xx = mxGetUint8s (X) ;
        }
        else if (type == GrB_UINT16)
        { 
            // full uint16 matrix
            Xx = mxGetUint16s (X) ;
        }
        else if (type == GrB_UINT32)
        { 
            // full uint32 matrix
            Xx = mxGetUint32s (X) ;
        }
        else if (type == GrB_UINT64)
        { 
            // full uint64 matrix
            Xx = mxGetUint64s (X) ;
        }
        else if (type == GrB_FP32)
        { 
            // full single matrix
            Xx = mxGetSingles (X) ;
        }
        else if (type == GxB_FC32)
        { 
            // full single complex matrix
            Xx = mxGetComplexSingles (X) ;
        }
        else
        {
            ERROR ("unsupported type") ;
        }

        if (X_is_sparse)
        { 
            // import the matrix in CSC format.  This sets Xp, Xi, and Xx to
            // NULL, but it does not change the MATLAB matrix they came from.
            OK (GxB_Matrix_import_CSC (&A, type, nrows, ncols,
                &Xp, &Xi, &Xx, ncols+1, nzmax, nzmax, false, NULL)) ;
        }
        else
        { 
            // import a full matrix
            OK (GxB_Matrix_import_FullC (&A, type, nrows, ncols, &Xx,
                nzmax, NULL)) ;
        }
    }

    //-------------------------------------------------------------------------
    // tell GraphBLAS the matrix is shallow
    //-------------------------------------------------------------------------

    // TODO need a shallow import
    A->p_shallow = (A->p != NULL) ;
    A->h_shallow = (A->h != NULL) ;
    A->b_shallow = (A->b != NULL) ;
    A->i_shallow = (A->i != NULL) ;
    A->x_shallow = (A->x != NULL) ;
    #ifdef GB_DEBUG
    if (A->p != NULL) GB_Global_memtable_remove (A->p) ;
    if (A->h != NULL) GB_Global_memtable_remove (A->h) ;
    if (A->b != NULL) GB_Global_memtable_remove (A->b) ;
    if (A->i != NULL) GB_Global_memtable_remove (A->i) ;
    if (A->x != NULL) GB_Global_memtable_remove (A->x) ;
    #endif

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    return (A) ;
}

