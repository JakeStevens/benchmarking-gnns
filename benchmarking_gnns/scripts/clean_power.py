#!/usr/bin/env python

import glob

for fname in glob.glob('power_results/*.power'):
    with open(fname, 'r') as f:
        trace = f.read()
        # Check to see if it is already cleaned
        if 'Average' in trace:
            print(trace)
            continue
        trace = [float(t) for t in trace.split()[2:] if t != 'W']
        first_third = len(trace)//3
        second_third = 2*len(trace)//3
        trace = trace[first_third:second_third]
        avg_power = sum(trace)/len(trace)
        print(f'{fname}: {avg_power}')
    with open(fname, 'w') as f:
        f.write(f'Average power: {avg_power}')

