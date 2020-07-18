//------------------------------------------------------------------------------
// GB_AxB_dot2_nomask:  C=A'*B via dot products
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{
    int ntasks = naslice * nbslice ;

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        int a_tid = tid / nbslice ;
        int b_tid = tid % nbslice ;

        //----------------------------------------------------------------------
        // get the counts of C for this slice of A
        //----------------------------------------------------------------------

        #if defined ( GB_PHASE_1_OF_2 )
        int64_t *GB_RESTRICT C_count = C_counts [a_tid] ;
        #else
        int64_t *GB_RESTRICT C_count_start =
            (a_tid == 0) ?         NULL : C_counts [a_tid] ;
        int64_t *GB_RESTRICT C_count_end   =
            (a_tid == naslice-1) ? NULL : C_counts [a_tid+1] ;
        #endif

        //----------------------------------------------------------------------
        // C=A'*B via dot products
        //----------------------------------------------------------------------

        for (int64_t kB = B_slice [b_tid] ; kB < B_slice [b_tid+1] ; kB++)
        {

            //------------------------------------------------------------------
            // get B(:,j)
            //------------------------------------------------------------------

            int64_t j = GBH (Bh, kB) ;
            int64_t pB_start = GBP (Bp, kB, bvlen) ;
            int64_t pB_end   = GBP (Bp, kB+1, bvlen) ;

            int64_t bjnz = pB_end - pB_start ;
            // no work to do if B(:,j) is empty
            if (bjnz == 0) continue ;

            //------------------------------------------------------------------
            // phase 2 of 2: get the range of entries in C(:,j) to compute
            //------------------------------------------------------------------

            #if defined ( GB_PHASE_2_OF_2 )
            // this thread computes Ci and Cx [cnz:cnz_last]
            int64_t cnz = Cp [kB] +
                ((C_count_start == NULL) ? 0 : C_count_start [kB]) ;
            int64_t cnz_last = (C_count_end == NULL) ?
                (Cp [kB+1] - 1) :
                (Cp [kB] + C_count_end [kB] - 1) ;
            if (cnz > cnz_last) continue ;
            #endif

            //------------------------------------------------------------------
            // C(:,j) = A'*B(:,j)
            //------------------------------------------------------------------

            // get the first and last index in B(:,j)
            int64_t ib_first = GBI (Bi, pB_start, bvlen) ;
            int64_t ib_last  = GBI (Bi, pB_end-1, bvlen) ; ;

            // for each vector A(:,i):
            for (int64_t kA = A_slice [a_tid] ; kA < A_slice [a_tid+1] ; kA++)
            {

                //--------------------------------------------------------------
                // get A(:,i)
                //--------------------------------------------------------------

                int64_t i = GBH (Ah, kA) ;
                int64_t pA     = GBP (Ap, kA, avlen) ;
                int64_t pA_end = GBP (Ap, kA+1, avlen) ;

                //--------------------------------------------------------------
                // C(i,j) = A(:,i)'*B(:,j)
                //--------------------------------------------------------------

                #include "GB_AxB_dot_cij.c"
            }
        }
    }
}

