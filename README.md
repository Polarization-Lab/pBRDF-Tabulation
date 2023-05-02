# pBRDF-Tabulation
Generates pbsdf files from sphere geometries for use in Mitsuba 3


There are three scripts contained in this repository, with sphere geometry and sphere pbrdf tabulation being the two primary ones. 

sphere geometry.py masks an imported sphere geometry through a radial mask and an intensity threshold, then calculates the Rusinkiewicz coordinates for each pixel within the mask.
These coordinate bins are saved and used in the followup script, sphere pbrdf tabulation.py.

sphere pbrdf tabulation.py takes the sphere geometries in Rusinkiewicz coordinates and separates the Mueller matrices of sphere cmmi measurements into these coordinate bins. 
Once the Mueller values across the measured spheres are contained in their coordinate bins, each bin's Mueller matrix is found through averaging.
The unpopulated coordinate spaces in the pBRDF are interpolated using Nearest Neighbor methods. 
The wavelength selections for the pBRDF are 360, 451, 524, 662, & 830nm. The outer wavelenghts (360nm & 830nm) Copy measurements from 451nm & 662nm respectively.

The last script in this repository is kept to verify the sphere coordinates were calculated correctly through comparing the python scripts results with results generated through Mathematica.
