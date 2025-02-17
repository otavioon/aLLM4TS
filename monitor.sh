#!/bin/bash
clear
for f in */*/log.txt; do 
    echo -e "\n\n\n"
    echo "------------------- $f ------------------"
    tail -n 2 "$f"
    echo
done

