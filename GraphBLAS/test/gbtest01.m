function gbtest01
%GBTEST01 test GrB.ver and GrB.version

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('v = ver (''matlab'')\n') ;
v = ver ('matlab') ;
display (v) ;

fprintf ('v = GrB.ver\n') ;
v = GrB.ver ;
display (v) ;

fprintf ('v = GrB.version\n') ;
v = GrB.version ;
display (v) ;

fprintf ('GrB.ver\n\n') ;
GrB.ver

