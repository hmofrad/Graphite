#!/bin/bash
TILING=$1
for i in {1..1000..1}
do 
#    for j in {1..20..2}
 #   do	    
        j=2
        ./2d  ${i} ${j} ${TILING}
        if [ $? -ne 1 ]
        then
            echo "Test (${i}, ${j}) failed. ($?)"
	    exit
        else
            echo "Test (${i}, ${j}) passed. ($?)"
        fi
  # done	
done
