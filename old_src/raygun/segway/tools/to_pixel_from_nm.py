#!/usr/bin/env python

import sys

command_line = False
nm = ''

if len(sys.argv) > 1:
    command_line = True
    nm = sys.argv[1:]
    nm = ' '.join(nm)

while True:

    if not command_line:
        try:
            nm = input(": ")
        except:
            exit(0)
    # print(nm)
    # print(len(nm))
    # if len(nm) != 3:
    nm = nm.split(' ')
    # print(nm)

    if len(nm) == 3:

        if len(nm[0].split(':')) == 2:
            for i in range(len(nm)):
                nm[i] = int(nm[i].strip(',').split(':')[0])
            print(nm[2]/4, nm[1]/4, nm[0]/40)
            print("%d, %d, %d" % (nm[2]/4, nm[1]/4, nm[0]/40))
        elif nm[0][0] == '[':
            nm[0] = nm[0][1:]
            nm[2] = nm[2][:-1]
            for i in range(len(nm)):
                nm[i] = int(nm[i].strip(','))
            print(nm[2]/4, nm[1]/4, nm[0]/40)
            print("%d, %d, %d" % (nm[2]/4, nm[1]/4, nm[0]/40))
        else:
            for i in range(len(nm)):
                nm[i] = int(nm[i].strip(','))
            print(nm[2]*40, nm[1]*4, nm[0]*4)
            print("%d, %d, %d" % (nm[2]*40, nm[1]*4, nm[0]*4))

    if command_line:
        exit(0)
