#!/usr/bin/python
#python3 create-multiconstraint-input.py --format=metis --exp=phase --cons=5 --part=part/M6.part instances/M6.graph

import argparse
import sys
import os
import random
import numpy as np
import math


insep = " "
outsep = " "
comment = '%'

def create_outfile_name(instance,exp,cons):
    delim = "."
    name, suf = instance.split(delim) if delim in instance else (instance, "")
    return name + '-' + exp + '-' + str(cons) + delim + suf 

def get_first_elem(l):
    return l[0]


# read part files: row i -> p of vertex i
def read_part_file(filename):
    #  return [int(x) for x in f]
    a=[]
    with open(filename) as f:
        for line in f:
            if line.startswith(comment):
                continue
            else:
                a.append(int(line))
        return a

# create random weight vectors (of cons_nb weights) for each part.
# Weights take values from 1-19.
def create_cons_list_random(cons_nb, part_nb):
    assert (cons_nb < part_nb )," number of constraints should be less than number of parts!"
    try:
        return [random.sample(range(1,19), cons_nb) for i in range(part_nb)]
    except ValueError:
        print('Sample size exceeded population size.')

# create random weight vectors (of cons_nb weights) for each part
# to simulate multi-phase applications.
# Each weight corresponds to a phase and their value indicate
# vertex activity or not (0 or 1).
def create_cons_list_phase(cons_nb, part_nb):
    assert (cons_nb < part_nb )," number of constraints should be less than number of parts!"
    # percentage of inactive vertices in each phase
    # TODO: give as input 
    inact = [0, 0.25, 0.45, 0.65, 0.85]
    phase = [math.ceil(int(i * part_nb)) for i in inact]
    a = [[1]* cons_nb for i in range(part_nb)]
    lcons = np.array(a)
    for i in range(cons_nb):
        cl= [col[i] for col in a]
        cl = np.array(cl)
        try:
            for j in [random.sample(range(0, part_nb), phase[i])]:
                cl[j] = 0
                lcons[:,i] = cl
        except ValueError:
            print('Sample size exceeded population size.')
    return lcons


# replace header in metis format to account for weight addition.
def create_header_metis(l,cons):
    if len(l) < 2:
        print("Header is missing: ", file=sys.stderr)
        sys.exit(1)
    if len(l) == 2:
        return str(l[0]) + outsep + str(l[1]) + outsep + '10' + outsep + str(cons)
    if len(l) == 3 and l[2] > 1:
        return str(l[0]) + outsep + str(l[1]) + outsep + str(l[2]) + outsep + str(cons)
    if len(l) == 3 and l[2] < 2:
        return str(l[0]) + outsep + str(l[1]) + outsep + str(l[2] + 10) + outsep + str(cons)
    elif len(l) == 4:
        return str(l[0]) + outsep + str(l[1]) + outsep + str(l[2]) + outsep + str(cons + l[3])

# create new input file of metis format with added weights.    
def create_input_metis(instance, part, cons, exp):
    newline = os.linesep # newline based on your OS.
    target = create_outfile_name(instance,exp,cons)
    try:
        with open(target, 'w') as tf:
            nbp = 16 # ??
            first = True
            i = 0
            try:
                with open(instance, 'r') as f:
                    for line in f:
                        if line.startswith(comment):
                            continue
                        elif first:
                            l = [int(i) for i in line.split(insep)]
                            nbvtxs = get_first_elem(l)
                            line = create_header_metis(l,cons)
                            tf.write(line + newline)
                            p = read_part_file(part)
                            nb_part = max(p) + 1
                            assert (nb_part <= nbp )," number of partitions should less than 16!"
                            assert (len(p) == nbvtxs )," number of vertices in partition is incorrect!"
                            if exp == 'random':
                                lcons = create_cons_list_random(cons, nb_part)
                            elif exp == 'phase':
                                lcons = create_cons_list_phase(cons, nb_part)
                            else:
                                print("Unknown format: ", format, file=sys.stderr)
                                sys.exit(1)
                            first = False
                        else:
                            str1 = insep.join(str(e) for e in lcons[p[i]-1])
                            i+=1
                            tf.write(str1 + outsep + line)
            except IOError as error:
                raise
    except IOError as error:
        raise
    print("Writing output in file: ", target)

# create new input file with added weights for different formats (metis,..).    
def create_multicons_input(format, exp, cons, instance, part):
    if format == 'metis':
        create_input_metis(instance,part,cons,exp)
    elif format == 'other':
        print("Not yet implemented: ", format, file=sys.stderr)
    else:
        print("Unknown format: ", format, file=sys.stderr)
        sys.exit(1)
        
def do_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', type=str, choices=['metis', 'other'])
    parser.add_argument('--exp', type=str, choices=['random', 'phase'])
    parser.add_argument('--cons', type=int, choices=[1,2,3,4,5])
    parser.add_argument('instance', type=str)
    parser.add_argument('--part', type=str)
    args = parser.parse_args()
    print(args)
    #for a in args:
    #    print(a)
    #print('arguments: format.graph exp.type nb.cons instance.graph part.graph')
    create_multicons_input(args.format, args.exp, args.cons, args.instance, args.part)
    
do_main()
