set terminal qt
set xlabel "x"
set ylabel "y"
set zlabel "times (us)"
set hidden3d offset 0  # offset 0: each surface have the same color on top side and bottom side
set dgrid3d 50,50 splines

# https://stackoverflow.com/questions/17120363/default-colour-set-on-gnuplot-website
# https://www2.uni-hamburg.de/Wiss/FB/15/Sustainability/schneider/gnuplot/colors.htm
set linetype  1 lc rgb "dark-violet" lw 1
set linetype  2 lc rgb "#009e73" lw 1
set linetype  3 lc rgb "#56b4e9" lw 1
set linetype  4 lc rgb "#e69f00" lw 1
set linetype  5 lc rgb "#f0e442" lw 1
set linetype  6 lc rgb "#0052f2" lw 1
set linetype  7 lc rgb "#e51e10" lw 1
set linetype  8 lc rgb "black"   lw 1
set linetype  9 lc rgb "gray50"  lw 1
set linetype  10 lc rgb "#ff1493"  lw 1
set linetype  11 lc rgb "orchid"  lw 1
set linetype cycle  11

splot 'binary.csv' using 1:2:3 with lines, 'gallop.csv' using 1:2:3 with lines, 'gallop2.csv' using 1:2:3 with lines, 'gallop3.csv' using 1:2:3 with lines, 'gallop_sse4.csv' using 1:2:3 with lines, 'linear_sse4.csv' using 1:2:3 with lines
