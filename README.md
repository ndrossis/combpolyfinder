# combpolyfinder
A machine learning algorithm to predict comb polymer monolayers that will have a chosen permeability on a given surface

# ABOUT
This python script loads a pre-trained feed forward neural network model and uses it to predict possible comb polymer monolayers that may yield a permiability similar to one that the user has specified. The dataset this FFNN was trained on was created by running a series of coarse grained molecular dynamics simulations with comb polymers grafted to a curved surface. Across simulations four variables were changed, surface curvature (R), grafting density (rho_graft), main-chain length (n) and side-chain length (m). Simulations were preformed for each case with probe particles of different sizes and the largest probe particle that was able to reach the surface was recorded for each case (r_p,max).

A simple FFNN was trained on this data, with 4 inputs (R, rho_graft, n and m) and 6 outputs (r_p,max, rho_bead, Rg_z, Rg_xy, n_colbead and A_empty), the model was composed of 2 hidden layers with 10 neurons, and used a sigmoid activation function. The mean squared error was used as the cost. The data was separated into a training set composed of 90% of the cases and a testing set composed of 10% of cases. The training set was then split into 8 batches and the model was trained on those batches for 80000 generations. After training, the model was run over the test data and the following root mean squared errors were calculated for each output:

|  rho_bead	|	Rg_z	    	|	Rg_xy	    	|	n_colbead	  |	Aempty		  |	r_p,max		  |	
|  0.017424	|	0.01035824	|	0.01151342	|	0.00900664	|	0.01645355	|	0.02409158	|

The python script included here loads that trained model, creates an array of possible values of rho_graft, n and m, and then runs the model over that array, finding any cases that give a value of r_p,max that are within +/- tolerance of the user specified target r_p,max.

# CONTENTS
alldata1.txt
- Data file containing all of the cases simulated and all of the quantities mesured for each of those cases. This was used as the testing/training dataset for the feed forward neural network.

poresize_model.meta
- The trained model.

finder.py
- The run script, it takes three inputs, surface radius, target permeability and error threshold, and returns a list of possible candidate cases with the target permeability as well as predictions of some of the other output parameters for each of those cases.

# USING THIS CODE
This code was written using python 3.10.12 and tensorflow 2.18.0
Run the code from your command line and it will prompt you to input the surface radius of the system you are interested in, the target r_p,max and the error acceptable in the output r_p,max. All of these are in simulation normalized units of sigma, where 1 sigma is the diameter of a single side-chain bead (for the example of pOEGMA, this would mean 1 sigma = 0.3 nm, but as these simulations are generalized, another comb polymer could be used, but be sure to use that conversion for your inputs (simply divide the radius of your protein by the diameter of one repeating side-chain unit of your comb polymer, and input that as your R, then do the same for your target r_p,max)). Then choose if you would like to limit your outputs to a specific range of grafting densities, and if applicable you will then be prompted to enter the range.

A list of cases will then be generated, the grafting density, main-chain length and side-chain length will be displayed as well as some output parameters:
rho_bead: an estimate of the polymer bead density around the surface
Rg_z: an estimate of the average radius of gyration of the comb polymers in the radial direction
Rg_xy: an estimate of the average radius of gyration of the comb polymers around Rg_z
n_colbead: an estimate of the average number of inter-chain collisions normalized by each bead (less than 0.15 indicates no inter-chain contact, between 0.15 and 1.5 indicates slight inter-chain contact, larger than 1.5 indicates strong inter-chain contact)
Aempty: an estimate of the amount of space on the surface unblocked by comb polymer beads
r_p,max: the estimated largest probe particle size that will be able to reach the surface for this case
