//------------------------------------------------------------------------------
// GB_transplant: replace contents of one matrix with another
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Transplant A into C, and then free A.  If any part of A is shallow, or if A
// must be typecasted, a deep copy is made into C.  Prior content of C is
// ignored.  Then A is freed, except for any shallow components of A which are
// left untouched (after unlinking them from A).  The resulting matrix C is not
// shallow.  This function is not user-callable.  The new type of C (ctype)
// must be compatible with A->type.

#include "GB.h"

GrB_Info GB_transplant          // transplant one matrix into another
(
    GrB_Matrix C,               // output matrix to overwrite with A
    const GrB_Type ctype,       // new type of C
    GrB_Matrix *Ahandle,        // input matrix to copy from and free
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Ahandle != NULL) ;
    GrB_Matrix A = *Ahandle ;
    ASSERT (!GB_aliased (C, A)) ;

    ASSERT (C != NULL) ;
    ASSERT_MATRIX_OK (A, "A before transplant", GB0) ;
    ASSERT_TYPE_OK (ctype, "new type for C", GB0) ;

    // pending tuples may not appear in A
    ASSERT (!GB_PENDING (A)) ;

    // zombies in A can be safely transplanted into C
    ASSERT (GB_ZOMBIES_OK (A)) ;

    // C is about to be cleared, so zombies and pending tuples are OK
    ASSERT (GB_PENDING_OK (C)) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;

    // the ctype and A->type must be compatible.  C->type is ignored
    ASSERT (GB_Type_compatible (ctype, A->type)) ;

    int64_t avdim = A->vdim ;
    int64_t avlen = A->vlen ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t anz = GB_NNZ (A) ;
    int64_t anvec = A->nvec ;

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (anz + anvec, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // clear C and transplant the type, size, and hypersparsity
    //--------------------------------------------------------------------------

    // free all content of C
    GB_phix_free (C) ;

    ASSERT (!GB_PENDING (C)) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (C->nzmax == 0) ;

    // It is now safe to change the type, dimension, and hypersparsity of C
    C->type = ctype ;
    C->is_csc = A->is_csc ;
    C->vlen = avlen ;
    C->vdim = avdim ;
    ASSERT (A->nvec_nonempty == -1 ||   // can be postponed
            A->nvec_nonempty == GB_nvec_nonempty (A, Context)) ;
    C->nvec_nonempty = A->nvec_nonempty ;

    // C->hyper_ratio is not modified by the transplant

    // C is not shallow, and has no content
    ASSERT (!C->p_shallow && !C->h_shallow && !C->i_shallow && !C->x_shallow) ;
    ASSERT (C->h == NULL && C->p == NULL && C->i == NULL && C->x == NULL) ;

    // determine if C should be constructed as a full matrix
    bool C_is_full = GB_is_dense (A) && !GB_ZOMBIES (A) ;

    //--------------------------------------------------------------------------
    // transplant A->p vector pointers and A->h hyperlist
    //--------------------------------------------------------------------------

    if (C_is_full)
    { 

        //----------------------------------------------------------------------
        // C is full
        //----------------------------------------------------------------------

        C->plen = -1 ;
        C->nvec = avdim ;

        // free any non-shallow A->p and A->h content of A
        GB_ph_free (A) ;

    }
    else if (A->p_shallow || A->h_shallow)
    {

        //----------------------------------------------------------------------
        // A->p or A->h are shallow copies another matrix; make a deep copy
        //----------------------------------------------------------------------

        int nth = GB_nthreads (anvec, chunk, nthreads_max) ;

        if (A->h != NULL)
        {
            // A is hypersparse, create new C->p and C->h
            C->plen = anvec ;
            C->nvec = anvec ;
            C->p = GB_MALLOC (C->plen+1, int64_t) ;
            C->h = GB_MALLOC (C->plen  , int64_t) ;
            if (C->p == NULL || C->h == NULL)
            { 
                // out of memory
                GB_phix_free (C) ;
                GB_Matrix_free (Ahandle) ;
                return (GrB_OUT_OF_MEMORY) ;
            }

            // copy A->p and A->h into the newly created C->p and C->h
            GB_memcpy (C->p, A->p, (anvec+1) * sizeof (int64_t), nth) ;
            GB_memcpy (C->h, A->h,  anvec    * sizeof (int64_t), nth) ;
        }
        else
        {
            // A is non-hypersparse, create new C->p
            C->plen = avdim ;
            C->nvec = avdim ;
            C->p = GB_MALLOC (C->plen+1, int64_t) ;
            if (C->p == NULL)
            { 
                // out of memory
                GB_phix_free (C) ;
                GB_Matrix_free (Ahandle) ;
                return (GrB_OUT_OF_MEMORY) ;
            }

            // copy A->p into the newly created C->p
            GB_memcpy (C->p, A->p, (avdim+1) * sizeof (int64_t), nth) ;
        }

        // free any non-shallow A->p and A->h content of A
        GB_ph_free (A) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // both A->p and A->h are not shallow: quick transplant into C
        //----------------------------------------------------------------------

        // Quick transplant of A->p and A->h into C.  This works for both
        // standard and hypersparse cases.
        ASSERT (C->p == NULL) ;
        ASSERT (C->h == NULL) ;
        C->p = A->p ;
        C->h = A->h ;
        C->plen = A->plen ;
        C->nvec = anvec ;
    }

    // A->p and A->h have been freed or removed from A
    A->p = NULL ;
    A->h = NULL ;
    A->p_shallow = false ;
    A->h_shallow = false ;
    C->p_shallow = false ;
    C->h_shallow = false ;

    C->magic = GB_MAGIC ;          // C is now initialized

    if (anz == 0)
    { 
        // quick return if A has no entries
        ASSERT_MATRIX_OK (C, "C empty transplant", GB0) ;
        GB_Matrix_free (Ahandle) ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // allocate new space for C->i and C->x if A is shallow
    //--------------------------------------------------------------------------

    // get C->nzmax:  if either C->x or C->i must be allocated, then C->nzmax
    // is set to their minimum size.  Otherwise, if both C->x and C->i can
    // be transplanted from A, then they inherit the nzmax of A.

    // Do not allocate C->i if C is full.

    ASSERT (C->x == NULL && C->i == NULL) ;
    bool allocate_Ci = (A->i_shallow) && (!C_is_full) ;
    bool allocate_Cx = (A->x_shallow || C->type != A->type) ;
    C->nzmax = (allocate_Cx || allocate_Ci) ? anz : A->nzmax ;
    C->nzmax = GB_IMAX (C->nzmax, 1) ;

    // allocate new components if needed
    bool ok = true ;
    if (allocate_Cx)
    { 
        // allocate new C->x component
        C->x = GB_MALLOC (C->nzmax * C->type->size, GB_void) ;
        ok = ok && (C->x != NULL) ;
    }

    if (allocate_Ci)
    { 
        // allocate new C->i component
        C->i = GB_MALLOC (C->nzmax, int64_t) ;
        ok = ok && (C->i != NULL) ;
    }

    if (!ok)
    { 
        // out of memory
        GB_phix_free (C) ;
        GB_Matrix_free (Ahandle) ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // transplant or copy A->x numerical values
    //--------------------------------------------------------------------------

    // Note that A may contain zombies, and the values of these zombies may be
    // uninitialized values in A->x.  All entries are typecasted or memcpy'ed
    // from A->x to C->x, both zombies and live entries alike.  valgrind may
    // complain about typecasting these uninitialized values, but these
    // warnings are false positives.  The output of the typecasting is itself a
    // zombie, and the values of all zombies are ignored.

    ASSERT_TYPE_OK (C->type, "target C->type for values", GB0) ;
    ASSERT_TYPE_OK (A->type, "source A->type for values", GB0) ;

    if (C->type == A->type)
    {
        // types match
        if (A->x_shallow)
        { 
            // A is shallow so make a deep copy; no typecast needed
            GB_memcpy (C->x, A->x, anz * C->type->size, nthreads) ;
            A->x = NULL ;
        }
        else
        { 
            // OK to move pointers instead
            C->x = A->x ;
            A->x = NULL ;
        }
    }
    else
    {
        // types differ, must typecast from A to C.
        GB_void *GB_RESTRICT Cx = (GB_void *) C->x ;
        GB_void *GB_RESTRICT Ax = (GB_void *) A->x ;
        GB_cast_array (Cx, C->type->code,
            Ax, A->type->code, A->type->size, anz, nthreads) ;
        if (!A->x_shallow)
        { 
            GB_FREE (A->x) ;
        }
        A->x = NULL ;
    }

    ASSERT (A->x == NULL) ;     // has been freed or removed
    A->x_shallow = false ;

    ASSERT (C->x != NULL) ;
    C->x_shallow = false ;

    //--------------------------------------------------------------------------
    // transplant or copy A->i row indices
    //--------------------------------------------------------------------------

    if (C_is_full)
    { 

        //----------------------------------------------------------------------
        // C is full
        //----------------------------------------------------------------------

        // C is dense; C->i stays NULL
        C->i = NULL ;

    }
    else if (A->i_shallow)
    {

        //----------------------------------------------------------------------
        // A->i is a shallow copy of another matrix, so we need a deep copy
        //----------------------------------------------------------------------

        // copy A->i into C->i
        GB_memcpy (C->i, A->i, anz * sizeof (int64_t), nthreads) ;
        A->i = NULL ;
        A->i_shallow = false ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // A->i is not shallow, so just transplant the pointer from A to C
        //----------------------------------------------------------------------

        C->i = A->i ;
        A->i = NULL ;
        A->i_shallow = false ;
    }

    C->i_shallow = false ;
    C->nzombies = A->nzombies ;     // zombies may have been transplanted into C

    //--------------------------------------------------------------------------
    // free A and return result
    //--------------------------------------------------------------------------

    GB_Matrix_free (Ahandle) ;
    ASSERT_MATRIX_OK (C, "C after transplant", GB0) ;
    return (GrB_SUCCESS) ;
}

