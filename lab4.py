import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind, f_oneway, f

class StdAdd:
	def __init__(self, x_data, y_data):
		assert(len(x_data) == len(y_data)), "Input data lists must match in length!"
		self.m = 0  #slope
		self.b = 0  #offset aka yint
		self.x_data = np.asarray(x_data)
		self.y_data = np.asarray(y_data)
		self.n = len(x_data)

	def compute_model(self, verbose = False):
		self.xy_sum = np.sum(self.x_data * self.y_data)
		self.x_sum = np.sum(self.x_data)
		self.y_sum = np.sum(self.y_data)
		self.x2_sum = np.sum(np.square(self.x_data))
		self.y2_sum = np.sum(np.square(self.y_data))
		self.m = ((self.n * self.xy_sum) - self.x_sum * self.y_sum) / (self.n * self.x2_sum - np.square(self.x_sum))
		self.b = ((self.x2_sum * self.y_sum) - self.x_sum * self.xy_sum) / (self.n * self.x2_sum - np.square(self.x_sum))
		r = (self.n * self.xy_sum - self.x_sum * self.y_sum) / np.sqrt((self.n * self.x2_sum - np.square(self.x_sum)) * (self.n * self.y2_sum - np.square(self.y_sum)))
		self.r2 = np.square(r)
		if verbose:
			print "Computing model...\n******************************"
			print "Slope: ", self.m
			print "Offset: ", self.b
			print "R2: " + str(self.r2)
			print "\n"

	def _prop_error(self):
		#Performs propagation of error for dilutions
		V0 = 1
		V1 = 25
		V2 = 4
		V3 = 25
		dV0 = 0.006
		dV1 = 0.12
		dV2 = 0.024
		dV3 = 0.12
		print "Error: ", self.error
		print "Term 1: " , np.square(V1 * V3 / float(V2 * V0)) * np.square(self.error)
		print "Term 2: ", np.square(self.c * V3 / float(V2 * V0)) * np.square(dV1)
		print "Term 3: ", np.square(self.c * V1 / float(V2 * V0)) * np.square(dV3)
		print "Term 4: ", np.square(self.c * V1 * V3 / float(np.square(V2) * V0)) * np.square(dV2)
		print "Term 5: ", np.square(self.c * V1 * V3/float(V2 * np.square(V0))) * np.square(dV0)
		self.prop_error = np.sqrt(np.square(V1 * V3 / float(V2 * V0)) * np.square(self.error) + np.square(self.c * V3 / float(V2 * V0)) * np.square(dV1) + np.square(self.c * V1 / float(V2 * V0)) * np.square(dV3) + np.square(self.c * V1 * V3 / float(np.square(V2) * V0)) * np.square(dV2) + np.square(self.c * V1 * V3/float(V2 * np.square(V0))) * np.square(dV0))

	def compute_unknown(self, verbose = False):
		self.c = -self.b/self.m
		self.x_bar = np.mean(self.x_data)
		self.y_bar = np.mean(self.y_data)
		self.Sxx = self.x2_sum - self.n * np.square(self.x_bar)
		self.Syy = self.y2_sum - self.n * np.square(self.y_bar)
		self.Sxy = self.xy_sum - self.n * np.square(self.x_bar * self.y_bar)
		self.sy = np.sqrt((self.Syy - np.square(self.m) * self.Sxx)/float(self.n - 2))
		self.error = np.sqrt(np.square(self.sy/self.m) * ((1/float(self.n)) + np.square(self.y_bar)/(np.square(self.m) * self.Sxx))) #error in extrapolation
		if verbose:
			print "Computing unknown concentration...\n******************************"

		multiplier = 156.25
		self._prop_error()
		print "Extrapolated value: " + str(self.c)
		print "Unknown concentration: " + str(self.c * multiplier)  + " +/- " + str(self.prop_error)
		print "\n"

	def plot(self, save_path):
		x_min = -self.c 
		x_max = np.amax(self.x_data)
		y_min = 0
		y_max = np.amax(self.y_data)
		X = np.linspace(x_min, x_max, 50)
		Y = (X - self.b)/self.m 
		plt.figure()
		plt.plot(X, Y, '-k')
		plt.scatter(self.x_data, self.y_data, c='b')
		plt.xlabel('[K+] Concentration (ppm)')
		plt.ylabel('Instrumental Response')
		plt.axis([x_min - 1, x_max + 1, y_min - 5, y_max + 5])
		plt.savefig(save_path, format = 'png')
		plt.close()

	def calculate_LOD(self):
		self.resid = []
		index = 0
		for x in self.x_data:
			y_hat = self.m * x + self.b
			self.resid.append(self.y_data[index] - y_hat)
			index = index + 1

		self.MSE_resid = np.sqrt(np.sum(np.square(self.resid)))
		self.LOD = 3.3 * self.MSE_resid / self.m 
		self.LOQ = 10 * self.MSE_resid / self.m
		print "Limit of Detection: ", self.LOD
		print "Limit of Quantification: ", self.LOQ

def ANOVA(m_list, std_list, n_list, verbose=False):
	#m_list = list of means
	#std_list = list of std devs
	#n_list = list of number of elements in each sample
	df1 = len(n_list) - 1
	m_list = np.asarray(m_list)
	std_list = np.asarray(std_list)
	n_list = np.asarray(n_list)
	df2 = np.sum(n_list) - df1 - 1
	x_hat = np.sum(n_list * m_list) / float(np.sum(n_list))
	MS_error = np.sum(n_list * np.square(std_list))/float(df2)
	MS_group = np.sum(n_list * np.square(m_list - x_hat)) / float(df1)
	F = MS_group / MS_error
	p = 1 - f.cdf(F, df1, df2)
	if verbose:
		print '\n\n'
		print 'ANOVA Summary:'
		print 'df1 =', df1
		print 'df2 =', df2
		print 'SS_group =', np.sum(n_list * np.square(m_list - x_hat))
		print 'SS_error =', np.sum(n_list * np.square(std_list))
		print 'MS_group =', MS_group
		print 'MS_error =', MS_error
		print 'F =', F
		print 'p-value =', p
	return F, p




if __name__ == '__main__':
	std1 = [5, 16.25, 27.5, 38.75, 50]
	IC1 = [0.3542, 0.6163, 0.7983, 0.5032, 0.7917]
	ICP1 = [4.468, 16.05, 28, 39.53, 51.1]

	std2 = [10, 17.5, 25, 32.5, 40]
	IC2 = [0.3709, 0.0722, 21.4928, 31.5402, 37.2712]
	ICP2 = [8.717, 16.63, 23.25, 33.15, 37.89]

	std3 = [10, 18.5971, 27.0462, 36.7847, 45.2261]
	IC3 = [1.1955, 0.9918, 1.0765, 7.1826, 2.4591]
	ICP3 = [9.674, 18.12, 28.07, 40.41, 44.29]

	std4 = [10, 16.62, 23.16, 29.61, 35.97]
	IC4 = [7.4095, 20.9944, 21.2314, 29.8502, 41.9995]
	ICP4 = [8.361, 22.76, 22.03, 31.46, 43.45]

	std5 = [5, 15, 25, 35, 45]
	IC5 = [3.9239, 14.0717, 23.9931, 34.4023, 44.5903]
	ICP5 = [4.591, 15.01, 25.03, 35.16, 45.43]

	std6 = [5, 15, 25, 35, 45]
	IC6 = [4.0255, 14.0041, 24.1484, 35.168, 45.4753]
	ICP6 = [4.643, 15.36, 25.54, 36.17, 46.19]

	std7 = [5, 10, 25, 40, 50]
	IC7 = [4.0409, 8.9792, 23.8228, 42.6273, 46.7795]
	ICP7 = [4.664, 9.602, 24.95, 40.21, 49.71]

	std8 = [5, 15, 25, 35, 45]
	IC8 = [3.9217, 14.5803, 24.1324, 35.4043, 46.6756]
	ICP8 = [4.71, 15.39, 25.68, 37.48, 47.6]

	std9 = [10, 19.9, 29.8, 39.7, 49.6]
	IC9 = [8.4107, 18.1918, 29.5948, 40.878, 49.4073]
	ICP9 = [9.192, 19.18, 30.21, 40.44, 49.07]

	std10 = [5, 15, 25, 35, 45]
	IC10 = [4.0854, 14.2788, 24.4306, 33.9463, 45.9165]
	ICP10 = [4.67, 14.94, 25.4, 35.99, 46.17]

	std11 = [8, 16, 24, 32, 40]
	IC11 = [6.4483, 14.603, 22.8055, 31.5856, 37.1986]
	ICP11 = [7.416, 15.5, 24.17, 32.56, 38.87]

	std_list = [std1, std2, std3, std4, std5, std6, std7, std8, std9, std10, std11]
	IC_list = [IC1, IC2, IC3, IC4, IC5, IC6, IC7, IC8, IC9, IC10, IC11]
	ICP_list = [ICP1, ICP2, ICP3, ICP4, ICP5, ICP6, ICP7, ICP8, ICP9, ICP10, ICP11] 

	IC_C = []
	ICP_C = []
	IC_std = []
	ICP_std = []
	IC_LOD = []
	IC_LOQ = []
	ICP_LOD = []
	ICP_LOQ = []

	multiplier = 156.25

	root_direc = "/Users/nicolasquach/Documents/stanford/senior_yr/spring/CHEM134/labs/lab4/plots/"

	for i in range(11):
		model_IC = StdAdd(std_list[i], IC_list[i])
		model_ICP = StdAdd(std_list[i], ICP_list[i])
		print "IC Data " + str(i+1) + ":"
		model_IC.compute_model(verbose=True)
		model_IC.compute_unknown()
		model_IC.calculate_LOD()
		filename = 'IC' + str(i + 1) + '.png'
		savepath = os.path.join(root_direc, filename)
		model_IC.plot(savepath)
		IC_C.append(model_IC.c)
		IC_std.append(model_IC.prop_error)
		IC_LOD.append(model_IC.LOD)
		IC_LOQ.append(model_IC.LOQ)
		print "\n\n"
		print "ICP Data " + str(i+1) + ":"
		model_ICP.compute_model(verbose=True)
		model_ICP.compute_unknown()
		model_ICP.calculate_LOD()
		filename = 'ICP' + str(i + 1) + '.png'
		savepath = os.path.join(root_direc, filename)
		model_ICP.plot(savepath)
		ICP_std.append(model_ICP.prop_error)
		ICP_C.append(model_ICP.c)
		print "\n\n"
		ICP_LOD.append(model_ICP.LOD)
		ICP_LOQ.append(model_ICP.LOQ)
		
	IC_C = IC_C[3:]  #Remove first data point which is an outlier
	IC_std = IC_std[3:]

	IC_LOD = np.asarray(IC_LOD[3:])# * multiplier
	IC_LOQ = np.asarray(IC_LOQ[3:]) #* multiplier

	IC_C = np.asarray(IC_C) * multiplier
	ICP_C = np.asarray(ICP_C) * multiplier

	std_IC = np.sqrt(np.mean(np.square(IC_std)))
	std_ICP = np.sqrt(np.mean(np.square(ICP_std)))
	print "IC Concentrations: ", IC_C
	print "ICP Concentrations: ", ICP_C

	print "IC Mean [K+]: ", np.mean(IC_C), " +/- ", std_IC
	print "ICP Mean [K+]: ", np.mean(ICP_C), " +/- ", std_ICP
	print "IC Mean LOD: ", np.mean(IC_LOD)
	print "IC Mean LOQ: ", np.mean(IC_LOQ)
	print "ICP Mean LOD: ", np.mean(ICP_LOD)
	print "ICP Mean LOQ: ", np.mean(ICP_LOQ)

	t, p = ttest_ind(IC_C, ICP_C, equal_var=False)
	print "t-Test of independence for concentrations (no error prop): "
	print "t = " + str(t)
	print "p = " + str(p) 

	F, p_F = f_oneway(IC_C, ICP_C)
	print "F-test of independence for concentrations (no error prop):"
	print "F = " + str(F)
	print "p = " + str(p_F)

	F2, p_F2 = ANOVA([np.mean(IC_C), np.mean(ICP_C)], [std_IC, std_ICP], [len(IC_C), len(ICP_C)], verbose=True)
	print "F-test of independence for concentrations (with error prop):"
	print "F = " + str(F2)
	print "p = " + str(p_F2)

	t_LOD, p_LOD = ttest_ind(IC_LOD, ICP_LOD, equal_var=False)
	print "t-Test of independence for LOD: "
	print "t = " + str(t_LOD)
	print "p = " + str(p_LOD) 

	t_LOQ, p_LOQ = ttest_ind(IC_LOQ, ICP_LOQ, equal_var=False)
	print "t-Test of independence for LOQ: "
	print "t = " + str(t_LOQ)
	print "p = " + str(p_LOQ) 


