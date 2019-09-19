#!/bin/bash
for i in {1..100..1}
do 
    for j in {1..20..2}
    do	    
       # j=1
        ./2d  ${i} ${j}
        if [ $? -ne 1 ]
        then
            echo "Test (${i}, ${j}) failed. ($?)"
	    exit
        else
            echo "Test (${i}, ${j}) passed. ($?)"
        fi
   done	
done
