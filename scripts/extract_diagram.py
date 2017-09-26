#!/usr/bin/env python

import re
import sys

DIAGRAM_PATTERN=re.compile(r'\[DIAG\] \[([^]]+)\] (\S+) (\S+) (\S+)')

# https://www2.uni-hamburg.de/Wiss/FB/15/Sustainability/schneider/gnuplot/colors.htm
COLORS=['coral', 'blueviolet', 'darkgreen', 'deeppink', 'goldenrod', 'olive', 'orchid', 'royalblue', 'salmon']

def main():
    algorithms = []
    current_algorithm = None
    out = None
    for line in sys.stdin:
        m = re.search(DIAGRAM_PATTERN, line)
        if m:
            algo = m.group(1)
            x = m.group(2)
            y = m.group(3)
            t = m.group(4)
            if algo != current_algorithm:
                current_algorithm = algo
                if out:
                    out.close()
                filename = algo + '.csv'
                out = open(filename, 'w')
                algorithms.append(algo)
            out.write('%s\t%s\t%s\n' % (x, y, t))

    # output gnuplot script
    splot = "splot " + ", ".join(["'" + algo + ".csv' using 1:2:3 with lines" for algo in algorithms])
    print(splot)


if __name__ == '__main__':
    main()
