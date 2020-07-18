//------------------------------------------------------------------------------
// GB_AxB_dot3_slice: slice the entries and vectors for C<M>=A'*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Constructs a set of tasks that slice the matrix C for C<M>=A'*B.  The matrix
// C has been allocated, and its pattern will be a copy of M, but with some
// entries possibly turned into zombies.  However, on input, the pattern C->i
// does yet not contain the indices, but the work required to compute each
// entry in the dot product C(i,j)<M(i,j)> = A(:,i)'*B(:,j).

// The strategy for slicing of C and M is like GB_ek_slice, for coarse tasks.
// These coarse tasks differ from the tasks generated by GB_ewise_slice,
// since they may start in the middle of a vector.  If a single entry C(i,j)
// is costly to compute, it is possible that it is placed by itself in a
// single coarse task.

// FUTURE:: Ultra-fine tasks could also be constructed, so that the computation
// of a single entry C(i,j) can be broken into multiple tasks.  The slice of
// A(:,i) and B(:,j) would use GB_slice_vector, where no mask would be used.

#define GB_FREE_WORK \
    GB_FREE (Coarse) ;

#define GB_FREE_ALL         \
{                           \
    GB_FREE_WORK ;          \
    GB_FREE (TaskList) ;    \
}

#include "GB_mxm.h"
#include "GB_ek_slice.h"

//------------------------------------------------------------------------------
// GB_AxB_dot3_slice
//------------------------------------------------------------------------------

GrB_Info GB_AxB_dot3_slice
(
    // output:
    GB_task_struct **p_TaskList,    // array of structs, of size max_ntasks
    int *p_max_ntasks,              // size of TaskList
    int *p_ntasks,                  // # of tasks constructed
    int *p_nthreads,                // # of threads to use
    // input:
    const GrB_Matrix C,             // matrix to slice
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (p_TaskList != NULL) ;
    ASSERT (p_max_ntasks != NULL) ;
    ASSERT (p_ntasks != NULL) ;
    ASSERT (p_nthreads != NULL) ;
    // ASSERT_MATRIX_OK (C, ...) cannot be done since C->i is the work need to
    // compute the entry, not the row index itself.

    // C is always constructed as sparse or hypersparse, not full, since it
    // must accomodate zombies
    ASSERT (!GB_IS_FULL (C)) ;

    (*p_TaskList  ) = NULL ;
    (*p_max_ntasks) = 0 ;
    (*p_ntasks    ) = 0 ;
    (*p_nthreads  ) = 1 ;

    //--------------------------------------------------------------------------
    // determine # of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // get C
    //--------------------------------------------------------------------------

    const int64_t *GB_RESTRICT Cp = C->p ;  // ok: C is sparse
    int64_t *GB_RESTRICT Cwork = C->i ;     // ok: C is sparse
    const int64_t cnvec = C->nvec ;
    const int64_t cvlen = C->vlen ;
    const int64_t cnz = GB_NNZ (C) ;

    //--------------------------------------------------------------------------
    // compute the cumulative sum of the work
    //--------------------------------------------------------------------------

    // FUTURE:: handle possible int64_t overflow

    GB_cumsum (Cwork, cnz, NULL, GB_nthreads (cnz, chunk, nthreads_max)) ;
    double total_work = (double) Cwork [cnz] ;

    //--------------------------------------------------------------------------
    // allocate the initial TaskList
    //--------------------------------------------------------------------------

    int64_t *GB_RESTRICT Coarse = NULL ;
    int ntasks1 = 0 ;
    int nthreads = GB_nthreads (total_work, chunk, nthreads_max) ;
    GB_task_struct *GB_RESTRICT TaskList = NULL ;
    int max_ntasks = 0 ;
    int ntasks = 0 ;
    int ntasks0 = (nthreads == 1) ? 1 : (32 * nthreads) ;
    GB_REALLOC_TASK_LIST (TaskList, ntasks0, max_ntasks) ;

    //--------------------------------------------------------------------------
    // check for quick return for a single task
    //--------------------------------------------------------------------------

    if (cnvec == 0 || ntasks0 == 1)
    { 
        // construct a single coarse task that does all the work
        TaskList [0].kfirst = 0 ;
        TaskList [0].klast  = cnvec-1 ;
        TaskList [0].pC = 0 ;
        TaskList [0].pC_end  = cnz ;
        (*p_TaskList  ) = TaskList ;
        (*p_max_ntasks) = max_ntasks ;
        (*p_ntasks    ) = (cnvec == 0) ? 0 : 1 ;
        (*p_nthreads  ) = 1 ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // determine # of threads and tasks
    //--------------------------------------------------------------------------

    double target_task_size = total_work / (double) (ntasks0) ;
    target_task_size = GB_IMAX (target_task_size, chunk) ;
    ntasks1 = total_work / target_task_size ;
    ntasks1 = GB_IMIN (ntasks1, cnz) ;
    ntasks1 = GB_IMAX (ntasks1, 1) ;

    //--------------------------------------------------------------------------
    // slice the work into coarse tasks
    //--------------------------------------------------------------------------

    if (!GB_pslice (&Coarse, Cwork, cnz, ntasks1))
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // construct all tasks, both coarse and fine
    //--------------------------------------------------------------------------

    for (int t = 0 ; t < ntasks1 ; t++)
    {

        //----------------------------------------------------------------------
        // coarse task operates on A (:, k:klast)
        //----------------------------------------------------------------------

        int64_t pfirst = Coarse [t] ;
        int64_t plast  = Coarse [t+1] - 1 ;

        if (pfirst <= plast)
        { 
            // find the first vector of the slice for task taskid: the
            // vector that owns the entry Ci [pfirst] and Cx [pfirst].
            int64_t kfirst = GB_search_for_vector (pfirst, Cp, 0, cnvec,
                cvlen) ;

            // find the last vector of the slice for task taskid: the
            // vector that owns the entry Ci [plast] and Cx [plast].
            int64_t klast = GB_search_for_vector (plast, Cp, kfirst, cnvec,
                cvlen) ;

            // construct a coarse task that computes Ci,Cx [pfirst:plast].
            // These entries appear in C(:,kfirst:klast), but this task does
            // not compute all of C(:,kfirst), but just the subset starting at
            // Ci,Cx [pstart].  The task computes all of the vectors
            // C(:,kfirst+1:klast-1).  The task computes only part of the last
            // vector, ending at Ci,Cx [pC_end-1] or Ci,Cx [plast].  This
            // slice strategy is the same as GB_ek_slice.

            GB_REALLOC_TASK_LIST (TaskList, ntasks + 1, max_ntasks) ;
            TaskList [ntasks].kfirst = kfirst ;
            TaskList [ntasks].klast  = klast ;
            ASSERT (kfirst <= klast) ;
            TaskList [ntasks].pC     = pfirst ;
            TaskList [ntasks].pC_end = plast + 1 ;
            ntasks++ ;

        }
        else
        { 
            // This task is empty, which means the coarse task that computes
            // C(i,j) is doing too much work.
            // FUTURE:: Use ultra-fine tasks here instead, and split the work
            // for computing the single dot product C(i,j) amongst several
            // ultra-fine tasks.
            ;
        }
    }

    ASSERT (ntasks <= max_ntasks) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORK ;
    (*p_TaskList  ) = TaskList ;
    (*p_max_ntasks) = max_ntasks ;
    (*p_ntasks    ) = ntasks ;
    (*p_nthreads  ) = nthreads ;
    return (GrB_SUCCESS) ;
}

