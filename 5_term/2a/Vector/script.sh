#!/bin/bash

gcc -pg -Wall VectorProduct.c -o VectorProduct
./VectorProduct
gprof ./VectorProduct