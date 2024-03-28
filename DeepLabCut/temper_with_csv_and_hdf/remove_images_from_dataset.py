# Author: Akira Kudo
# Displays images belonging to a dataset to facilitate controlled removal of data from 
# an already labeled DLC dataset.

import matplotlib.pyplot as plt

# read csv / h5 to find the data
# for every image in the list:
#   show the image using plt
#   prompt whether it is wall rearing, grooming or something else
#   if wall rearing / grooming, prompt whether to keep or not
#   if kept, increase the number of kept wall rearing / groomings
#   at the end, store list of images to keep into a csv


