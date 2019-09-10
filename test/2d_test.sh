#!/bin/bash
#for i in {1..5}
#do
#i=10
#echo "Hello"
#doneo
echo "Bash version ${BASH_VERSION}..."
for i in {1..10..1}
    do 
        echo "Test $i:"
	./2d ${i} 1
    done
