#!/bin/bash
#for i in {1..5}
#do
#i=10
#echo "Hello"
#doneo
TILING=$1
echo "Tiling type test for ${TILING}..."
for i in {2..1000..1}
do 
    ./2d ${TILING}  ${i} 1
    if [ $? -ne 1 ]
    then
	    echo "Test $i failed. ($?)"
	    #exit
    else
	    echo "Test $i passed. ($?)"
    fi	

done
