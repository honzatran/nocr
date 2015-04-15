#!/usr/bin/python
import sys

def parseFile( file_name ):
    f = open( file_name, 'r' )
    content = f.readlines()

    for line in content:
        parseLine(line.rstrip()) 

def parseLine( line ):
    l = line.split(" ")
    length = len(l)
    output = l[-1]
    print(output),
    for idx, val in enumerate(l[0:length-1]):
        print "{0}:{1}".format(idx +1 , val),
    print


def main():
    parseFile( sys.argv[1] )

main()
    
    

