set terminal qt
set xlabel "x"
set ylabel "y"
set zlabel "times (us)"
set hidden3d
set dgrid3d 50,50 splines

#splot 'query_result.csv' using 3:4:5 with lines, 'check_result.csv' using 3:4:5 with lines
#splot 'gallop2.csv' using 1:2:3 with lines, 'avx.csv' using 1:2:3 with lines, 'simd.csv' using 1:2:3 with lines
splot 'avx.csv' using 1:2:3 with lines, 'simd.csv' using 1:2:3 with lines
