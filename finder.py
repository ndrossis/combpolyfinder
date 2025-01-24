import tensorflow.compat.v1 as tf 
# disabling eager mode 
tf.compat.v1.disable_eager_execution() 
import numpy as np
import copy
import sys

# Open dataset for finding maxes to renormalize
rawdat1 = np.loadtxt('alldata1.txt',skiprows=1)

# Get user inputs
Rpro = input('\nEnter your surface radius:\n')
Rpro = float(Rpro)/max(rawdat1[:,0]) # Protein radius input
rpore_targ = -1
while rpore_targ < 0 or rpore_targ > 1:
	rpore_targ = input('\nEnter your target for r_p,max (between 0 and 11):\n')
	rpore_targ = float(rpore_targ)/11 # Target permiability
tolerance = input('\nEnter the tolerance (error +/-) to display cases within (recommended: ~0.25):\n')
tolerance = float(tolerance) #Error tolerance for suggested parameters
rho_graft_check = input('\nWould you like to restrict the values of rho_graft displayed (y/n)?\n')
rho_max = 0
rho_min = 0
if rho_graft_check == 'y':
	while rho_min < 0.002 or rho_min > 0.026:
		rho_min = input('\nEnter the minimum value of rho_graft (between 0.002 and 0.0255):\n')
		rho_min = float(rho_min)
	while rho_max < rho_min or rho_max > 0.026:
		rho_max = input('\nEnter the maximum value of rho_graft (between '+repr(rho_min)+' and 0.0255):\n')
		rho_max = float(rho_max)

# Create array of parameter space spanning rho_graft, n and m to be sampled over
in_data = np.array([[Rpro,min(rawdat1[:,1])/max(rawdat1[:,1]),1./max(rawdat1[:,2]),0]])
for rho in np.linspace(min(rawdat1[:,1])/max(rawdat1[:,1]),1,50):
	for n in np.linspace(1./10,1,10):
		for m in np.linspace(1./10,1,10):
			in_data = np.append(in_data,[[Rpro,rho,n,m]],axis=0)

# Rescaler for R, rho_graft, n and m
rescaler = [max(rawdat1[:,0]),max(rawdat1[:,1])/10000,max(rawdat1[:,2]),max(rawdat1[:,3])]

sess=tf.Session()    
# Load saved model
saver = tf.train.import_meta_graph('poresize_model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))
 
 

# Get the graph, input and output
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
output_layer = graph.get_tensor_by_name("op_lay:0")

# Run over the parameter space
yfit = sess.run([output_layer], feed_dict = {x: in_data})
# Find where in the dataset we are within 0.1% error of the target
porecolumn = [yfit[0][:,5]]
# Display suggested parameters and calculate outputs
success = 0
print("rho_graft	n	m	|	rho_bead	Rg_z	Rg_xy		n_colbead	Aempty		|	r_p,max")
for i in np.where(abs(np.array(porecolumn)*11-rpore_targ*11)<tolerance)[1]:
	if rho_graft_check == 'y':
		if in_data[i,1]*max(rawdat1[:,1])/10000 > rho_min and in_data[i,1]*max(rawdat1[:,1])/10000 < rho_max:
			print(repr(round(in_data[i,1]*max(rawdat1[:,1])/10000,5))+'		'+repr(int(in_data[i,2]*max(rawdat1[:,2])))+'	'+repr(int(in_data[i,3]*max(rawdat1[:,3])))+'	|	'+repr(round(yfit[0][i,0]*max(rawdat1[:,24]),4))+'		'+repr(round(yfit[0][i,1]*max(rawdat1[:,12]),4))+'	'+repr(round(yfit[0][i,2]*max(rawdat1[:,13]),4))+'		'+repr(round(yfit[0][i,3]*max(rawdat1[:,15]),4))+'		'+repr(round(yfit[0][i,4]*max(rawdat1[:,19]),4))+'		|	'+repr(round(yfit[0][i,5]*11,5)))
			success+=1
	else:
		print(repr(round(in_data[i,1]*max(rawdat1[:,1])/10000,5))+'		'+repr(int(in_data[i,2]*max(rawdat1[:,2])))+'	'+repr(int(in_data[i,3]*max(rawdat1[:,3])))+'	|	'+repr(round(yfit[0][i,0]*max(rawdat1[:,24]),4))+'		'+repr(round(yfit[0][i,1]*max(rawdat1[:,12]),4))+'	'+repr(round(yfit[0][i,2]*max(rawdat1[:,13]),4))+'		'+repr(round(yfit[0][i,3]*max(rawdat1[:,15]),4))+'		'+repr(round(yfit[0][i,4]*max(rawdat1[:,19]),4))+'		|	'+repr(round(yfit[0][i,5]*11,5)))
		success+=1
if success == 0:
	print('No matches found')