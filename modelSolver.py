#Analiza dopustnih resitev bioloskih modelov
#Copyright (C) 2018  Ziga Pusnik

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation version 3 of the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.


import random
import math
import numpy as np
import numpy.fft as fft
import peakutils
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint 
from scipy.cluster import hierarchy
from scipy.spatial.distance import cdist 
from sklearn import metrics, decomposition, preprocessing  
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans 
from deap import creator, base, tools, algorithms 
from copy import deepcopy   
import sys, traceback 
import pickle 
from scipy.interpolate import interp2d  
import locale

#plot config
locale.setlocale(locale.LC_ALL, "slovenian")       
mpl.rcParams['axes.formatter.use_locale'] = True       

#range of paremeter values	
parameter_values = {  "transcription": {"min": (0.02)*0.001, "max": (0.02)*1000, "ref": (0.02)}, 
				"translation": {"min": (0.075)*0.001, "max": (0.075)*1000, "ref": (0.075)}, 
				"protein_production": {"min": (0.75)*0.001, "max": (0.75)*1000, "ref": (7.5)},   
				"rna_degradation": {"min": (2)*0.01, "max": (2)*100, "ref": (2)},   
				"protein_degradation": {"min": (0.1)*0.01, "max": (0.1)*100, "ref": (0.1)},   
				"transport": {"min": (0.3)*0.01, "max": (0.3)*100, "ref": (0.3)},    
				"hill": {"min": (1)*0.1, "max": (1)*100, "ref": (1)},  
				"Kd": {"min": (1)*0.01, "max": (1)*100, "ref": (1)},          
				}   		

#simulated annealing				
class annealingSolver:   
	#class variables 
	fitSubjects = [] 
	T = 48
	dt = 0.001  
	N = T/dt  
	temp = 1000
	cost_fall = [] 
	current_cost = 0  
	
	#constructor
	def __init__(self, model, Y0, params, minCost, NGEN=100, populationSize = 100, gama=0.95, maxRep = 20):      
		self.model          	= model 
		self.Y0             	= Y0 
		self.gama 				= gama  
		self.NGEN           	= NGEN   
		self.minCost        	= minCost 
		self.populationSize 	= populationSize  
		self.nParams        	= params.size  
		self.params       		= params 
		self.currents 			= self.generateIndividuals()
		self.maxRep				= maxRep     
		self.sample_rate 		= 0.0033333333 #Hz 
		self.samples_per_hour 	= (1/self.dt)
		self.jump 				= int(self.samples_per_hour/(self.sample_rate*3600))
		self.cost_fall			= [0]*self.NGEN
		self.starting_ind 		= [deepcopy(x) for x in self.currents]
		self.per = self.T/4 
		self.amp = 100 
		self.CLK = [self.amp*(np.sin(2*math.pi*(x)/self.per) + 1)/2 for x in np.linspace(0, self.T, self.N*2)] 
	
	#generates an array of individuals
	def generateIndividuals(self):
		populationArray = [] 
		for i in range(self.populationSize): 
			randomArray = []
			for ind in range(self.nParams):
				randomArray.append(random.uniform(parameter_values[self.params[ind]]["min"], parameter_values[self.params[ind]]["max"]))
			populationArray.append(randomArray) 
		return populationArray 
		
	#evaluates individual	
	def evalIndividual(self, individual, plot=False):  
		#initial conditions
		y0 = self.Y0
		#time points
		ts = np.linspace(0, self.T, self.N)
		
		#flip flop model 
		if self.model == flipFlopModel :
			Y = [y0]
			y = y0
			t = 0 
			h = self.dt/2
			
			#Runge-Kutta method
			for j in range(0, self.N-1):
				i = 2*j
				nextCLK = self.CLK[i + 1]
				k1 = self.model(y, t, individual, self.CLK[i]) 
				k2 = self.model(y + k1*h, t + h, individual, nextCLK) 
				k3 = self.model(y + k2*h, t + h, individual, nextCLK)
				k4 = self.model(y + k3*self.dt, t + self.dt, individual, self.CLK[i + 2]) 
				
				y = y + ((k1 + 2*k2 + 2*k3 + k4)/6)*self.dt
				Y.append(y)  
				t = t + self.dt  	 
				
			Y = np.array(Y)  
			
			p1 = Y[:,2] 
			#1 sample per 5 minutes
			p1 = p1[0::self.jump]				
				
			sums = [(x-y)**2 for x, y in zip(p1, self.ideal)]
			cost = -np.sum(sums)

			return cost,
			
		#represilator model 
		try:
			#LSODA solver
			Y = odeint(self.model, y0, ts, args=(individual,))	
		except TypeError:
			return 0,	 	
				
		p1 = Y[:,0] 
		#fft sample rate: 1 sample per 5 minutes
		p1 = p1[0::self.jump] 
		
		fftData = abs(fft.rfft(p1))
		fftData = fftData[0:25]  
		fftData = np.array(fftData)
		indexes = peakutils.indexes(fftData, thres=0.02/max(fftData), min_dist=1)
		if len(indexes) == 0:
			return 0,
		
		indexes = peakutils.indexes(fftData, thres=0.02/max(fftData), min_dist=1)
		fitSamples = fftData[indexes] 
		spikeNum = len(indexes)
		std = getSTD(indexes, fftData, 1)
		diff = getDif(indexes, fftData)
		cost = std + diff
		 
		if cost > self.minCost:
			self.fitSubjects.append(individual)
		
		return cost,

	#get random neighbour
	def getNeighbour(self, subject, delta=0.2):   
		#select neighbour at random 
		neighbours = []
		multiplier = random.choice([-1, 1])
		ind = random.choice(range(self.nParams))
		neighbour = deepcopy(subject)
		neighbour[ind] = neighbour[ind] + neighbour[ind]*delta*multiplier
		if neighbour[ind] < parameter_values[self.params[ind]]["min"]:
			neighbour[ind] = parameter_values[self.params[ind]]["min"]
		elif neighbour[ind] > parameter_values[self.params[ind]]["max"]:
			neighbour[ind] = parameter_values[self.params[ind]]["max"]
					
		return neighbour
		
	#run parameter space search
	def run(self, temp, delta):
		self.temp = temp 
		self.currents = [deepcopy(x) for x in self.starting_ind]
		self.cost_fall	=  [0]*self.NGEN
		
		for i in range(self.populationSize):
			print(i)
			currRep = 0
			self.cost = self.evalIndividual(self.currents[i])[0]  
			for ind in range(self.NGEN):
				current = self.currents[i]
				neighbour = self.getNeighbour(current, delta)
				nextCost = self.evalIndividual(neighbour)[0]				
				
				#if got stuck in local optima 
				if currRep > self.maxRep:
					for k in range(ind, self.NGEN): 
						self.cost_fall[k] = self.cost_fall[k] + self.cost 
					break 
					
				self.cost_fall[ind] = self.cost_fall[ind] +  self.cost 
				
				if nextCost >= self.cost:  
					self.cost = nextCost
					self.currents[i] = neighbour
					currRep = 0
				else:
					P = math.exp(-((self.cost - nextCost)/(self.temp)))
					p = random.uniform(0, 1) 
					if p < P: 
						self.cost = nextCost
						self.currents[i] = neighbour
						currRep = 0  
					else:
						currRep = currRep + 1
					
					#temperature annealing
					self.temp = self.temp*self.gama 
		
		self.cost_fall = [x/self.populationSize for x  in self.cost_fall]   

#Genetic algorithm		
class geneticSolver:    
	global parameter_values 

	#constructor
	def __init__(self, model, Y0, params, populationSize = 250, indpb = 0.75, NGEN=25, minCost = 1e5, dt = 0.001 ):  
		self.T 					= 48	
		self.cost_fall 			= [] 
		self.current_cost 		= 0 	
		self.fitSubjects 		= []
		self.dt 				= dt
		self.model          	= model
		self.Y0           	  	= Y0  
		self.NGEN           	= NGEN 
		self.minCost        	= minCost 
		self.populationSize 	= populationSize
		self.nParams        	= params.size 
		self.params         	= params
		self.N              	= int(self.T/self.dt)
		self.sample_rate 		= 0.003333333333 #[Hz] one sample every five minutes
		self.samples_per_hour 	= (1/self.dt) 
		self.jump 				= int(self.samples_per_hour/(self.sample_rate*3600))
		self.indpb 				= indpb   
		self.per = self.T/4 
		self.amp = 100
		self.CLK = [self.amp*(np.sin(2*math.pi*(x)/self.per) + 1)/2 for x in np.linspace(0, self.T, self.N*2)] 
		self.ideal = [0]*self.N 
		self.ideal[0:int(self.N /4)] = [self.amp]*int(self.N /4) 
		self.ideal[2*int(self.N /4):3*int(self.N/4)] = [self.amp]*int(self.N /4) 
		self.ideal = self.ideal[0::self.jump] 
		creator.create("FitnessMax", base.Fitness, weights=(1.0,))
		creator.create("Individual", list, fitness=creator.FitnessMax)
		self.toolbox = base.Toolbox()	 
		self.toolbox.register("individual", self.generateIndividual)
		self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
		self.toolbox.register("evaluate", self.evalIndividual)
		self.toolbox.register("mate", tools.cxTwoPoint)
		self.toolbox.register("mutate", self.mutateIndividual, indpb=self.indpb, mult=2) 
		self.toolbox.register("select", tools.selTournament, tournsize=int(populationSize/4)) 
	
	#creates an array of random individuals
	def generateIndividual(self): 
		global parameter_values
		randomArray = []
		for ind in range(self.nParams):
			randomArray.append(random.uniform(parameter_values[self.params[ind]]["min"], parameter_values[self.params[ind]]["max"]))
		return creator.Individual(randomArray)

	#evaluates an individual 
	def evalIndividual(self, individual, plot=False):   
		#initial conditions    
		y0 = self.Y0
		#time points
		ts = np.linspace(0, self.T, self.N)   

		#flip flop model 
		if self.model == flipFlopModel: 
			Y = [y0]
			y = y0
			t = 0 
			h = self.dt/2
			
			#Runge Kutta method 
			for j in range(0, self.N-1):
				i = 2*j
				nextCLK = self.CLK[i + 1]
				k1 = self.model(y, t, individual, self.CLK[i]) 
				k2 = self.model(y + k1*h, t + h, individual, nextCLK) 
				k3 = self.model(y + k2*h, t + h, individual, nextCLK)
				k4 = self.model(y + k3*self.dt, t + self.dt, individual, self.CLK[i + 2]) 
				
				y = y + ((k1 + 2*k2 + 2*k3 + k4)/6)*self.dt
				Y.append(y)  
				t = t + self.dt 	 
				
			Y = np.array(Y) 
			
			p1 = Y[:,2] 
			#1 sample per 5 minutes
			p1 = p1[0::self.jump] 				
			sums = [(x-y)**2 for x, y in zip(p1, self.ideal)]
			cost = -np.sum(sums)
						
			if cost > self.minCost: 
				
				#check if is individual unique
				inArray = False
				for sub in self.fitSubjects:
					ind = np.array(individual)
					if (sub==individual).all(): 
						inArray = True
						break 
				#if individual is unique pickle it to file	
				if not inArray:
					self.file = open("results_flipflop_genetic", "ab")    
					#calculates amplitude and periode
					evalBy1 = p1[0:int(len(p1)/2)]  
					evalBy2 = p1[int(len(p1)/2):]  					
					peak_indexesMaks = peakutils.indexes(evalBy1, thres=0.02/max(evalBy1), min_dist=1) 
					peak_indexesMaksHalf = peakutils.indexes(evalBy2, thres=0.02/max(evalBy2), min_dist=1)  
					amp = 0 
					per = 0 
					if peak_indexesMaks.size and peak_indexesMaksHalf.size:
						fitSamples = evalBy1[peak_indexesMaks] 
						mean_maks = np.mean(fitSamples) 
						amp = mean_maks
						per = (peak_indexesMaksHalf[0] + int(len(p1)/2) - peak_indexesMaks[0])/12.0  

					#write individual to file	
					print("writting individual to file")    
					pickle.dump((np.array(individual), cost, amp, per), self.file)
					self.fitSubjects.append(np.array(individual))
					self.file.close()						
						
			self.current_cost = self.current_cost + -1*cost  		
			return cost,  

		#represilator model 
		try:
			#LSODA solver
			Y = odeint(self.model, y0, ts, args=(individual,))	 
		except TypeError:
			return 0, 


		p1 = Y[:,1]  
		#fft sample rate: 1 sample per 5 minutes
		p1 = p1[0::self.jump]  
		
		ts = np.linspace(0, self.T, self.N)	 	
		fftData = abs(fft.rfft(p1))
		fftData = fftData[0:25]  
		fftData = np.array(fftData) 
		indexes = peakutils.indexes(fftData, thres=0.02/max(fftData), min_dist=1) 
		
		if len(indexes) == 0:  
			return 0,  
		
		fitSamples = fftData[indexes]  
		spikeNum = len(indexes)	 	
		

		std = getSTD(indexes, fftData, 1)
		diff = getDif(indexes, fftData)
		cost = std + diff
		cost = cost*spikeNum 

		if cost > self.minCost:
	
			inArray = False
			for sub in self.fitSubjects:
				ind = np.array(individual)
				if (sub==individual).all():
					inArray = True
					break 
					
			if not inArray:
				self.file = open("results_rep_genetic", "ab") 
				#calculates amplitude and periode of the signal 
				evalBy = p1 
				evalByNeg = [-x for x in evalBy] 
				peak_indexesMaks = peakutils.indexes(evalBy, thres=0.02/max(evalBy), min_dist=1)
				peak_indexesMin = peakutils.indexes(evalByNeg, thres=0.02/max(evalByNeg), min_dist=1)
				amp = 0
				per = 0
				if peak_indexesMaks.size and peak_indexesMin.size:
				
					fitSamples = evalBy[peak_indexesMaks]  
					mean_maks = np.mean(fitSamples)
					fitSamples = evalBy[peak_indexesMin]  
					mean_min = np.mean(fitSamples)  
					amp = mean_maks - mean_min
					per = np.mean([2*np.abs(x - y) for x, y in zip(peak_indexesMin, peak_indexesMaks)])/12.0    
				
				print("writing individual to file") 
				#write individual to file 
				pickle.dump((np.array(individual), cost, amp, per), self.file)
				self.fitSubjects.append(np.array(individual))
				self.file.close() 	
		
		self.current_cost = self.current_cost + cost   
		return cost,
		
	#returns a tuple of mutated individual	
	def mutateIndividual(self, individual, indpb, mult): 	
		global parameter_values
		
		for idx, val in enumerate(individual):	
			rnd = random.uniform(0, 1)
			if rnd >= indpb:
				rnd2 = random.uniform(1 - mult, 1 + mult) 
				#print(rnd2)   
				individual[idx] = val*rnd2
				
				if individual[idx] < parameter_values[self.params[idx]]["min"]:
					individual[idx] = parameter_values[self.params[idx]]["min"] 
				elif individual[idx] > parameter_values[self.params[idx]]["max"]: 
					individual[idx] = parameter_values[self.params[idx]]["max"] 
					
		return individual,   
		
	#performs model simulations and plot it 
	def plotModel(self, subject, subtitle, plot=False): 
	
		ts = np.linspace(0, self.T, self.N)
		y0 = self.Y0
		
		if self.model == flipFlopModel: 
			Y = [y0]
			y = y0
			t = 0 
			h = self.dt/2 
			
			#Runge-Kutta method  
			for j in range(0, self.N-1):
				i = 2*j
				nextCLK = self.CLK[i + 1]
				k1 = self.model(y, t, subject, self.CLK[i]) 
				k2 = self.model(y + k1*h, t + h, subject, nextCLK) 
				k3 = self.model(y + k2*h, t + h, subject, nextCLK)
				k4 = self.model(y + k3*self.dt, t + self.dt, subject, self.CLK[i + 2]) 
				
				y = y + ((k1 + 2*k2 + 2*k3 + k4)/6)*self.dt
				Y.append(y)  
				t = t + self.dt 	 
				
			Y = np.array(Y) 
			p1 = self.CLK[0::2]
			p2 = Y[:,2]   
			p3 = Y[:,3]    
			
		else:	
			#LSODA solver
			Y = odeint(self.model, y0, ts, args=(subject,)) 
			p1 = Y[:,1]
			p2 = Y[:,3] 
			p3 = Y[:,5]  

			
		lines = plt.plot(ts, p1, 'r', ts, p2, 'g', ts, p3, 'b')
		plt.setp(lines[0], linewidth=1.5)
		plt.setp(lines[1], linewidth=1.5)
		plt.setp(lines[2], linewidth=1.5) 
		plt.ylabel('Količina proteinov [nM]')
		plt.xlabel(r"Čas [h]" + "\n" + "\n" + subtitle)  
		plt.legend(('$CLK$', '$q$', '$q_{c}$'), loc='upper right') 
		plt.legend(('X', 'Y', 'Z'), loc='upper right')      
		
		if plot:
			plt.show() 		
	
	#run parameter space search	
	def run(self): 
		#initialize new random population
		self.population = self.toolbox.population(self.populationSize) 	 
		self.cost_fall = [] 
		
		for gen in range(self.NGEN): 
			print(gen)
			#generate offspprings with crossover and mutations
			offspring = algorithms.varAnd(self.population, self.toolbox, cxpb=0.2, mutpb=0.2) 
			#evaluate individuals
			fits = self.toolbox.map(self.toolbox.evaluate, offspring)
			for fit, ind in zip(fits, offspring):
				ind.fitness.values = fit
			#roulete wheel selection
			self.population = self.toolbox.select(offspring, k=len(self.population))
			self.current_cost = self.current_cost/self.populationSize
			self.cost_fall.append(self.current_cost) 
			self.current_cost = 0  
		
		
#deterministc repressilator model
def repressilatorModel(Y, t, ind):
	#parameters are defined by individual
	alpha = ind[0]
	alpha0 = ind[1]
	n = ind[2]
	beta = ind[3]
	deltaRNA = ind[4]
	deltaP = ind[5]
	kd = ind[6] 
	mx = Y.item(0)
	my = Y.item(2) 
	mz = Y.item(4)
	x = Y.item(1)
	y = Y.item(3) 
	z = Y.item(5) 
	
	#in case of math range error
	try:
		dmx = -deltaRNA*mx + alpha/(1 + math.pow((z/kd), n)) + alpha0
		dmy = -deltaRNA*my + alpha/(1 + math.pow((x/kd), n)) + alpha0
		dmz = -deltaRNA*mz + alpha/(1 + math.pow((y/kd), n)) + alpha0
	except (OverflowError, ValueError):
		dmx = -deltaRNA*mx + alpha + alpha0
		dmy = -deltaRNA*my + alpha + alpha0
		dmz = -deltaRNA*mz + alpha + alpha0 
		
	dpx = beta*mx - deltaP*x
	dpy = beta*my - deltaP*y 
	dpz = beta*mz - deltaP*z
	
	return np.array([dmx, dpx, dmy, dpy, dmz, dpz])

#deterministic D flip flop model
def flipFlopModel(Y, t, ind, clk):
	a     = Y.item(0)
	not_a = Y.item(1)
	q     = Y.item(2)
	not_q = Y.item(3)
	d = not_q
	
	alpha1 = ind[0]
	alpha2 = ind[1]
	Kd1 =  ind[2]
	Kd2 = ind[3]
	Kd3 = ind[4]
	KdA = ind[5]
	delta = ind[6]

	alpha3 = ind[7]
	alpha4 = ind[8]
	Kd4 = ind[9]
	Kd5 = ind[10]
	Kd6 = ind[11]
	delta2 = ind[12]
	
	da_dt     = alpha1*heaviside(d-Kd1)*heaviside(Kd2-clk) + alpha2*heaviside(Kd3 - not_a) - delta*a
	dnot_a_dt = alpha1*heaviside(Kd1 - d)*heaviside(Kd2-clk) + alpha2*heaviside(Kd3 - a) - delta*not_a
	dq_dt     = alpha3*heaviside(a-Kd4)*heaviside(clk - Kd5)*heaviside(KdA - q) + alpha4*heaviside(Kd6 - not_q)*heaviside(KdA - q) - delta2*q
	dnot_q_dt = alpha3*heaviside(not_a - Kd4)*heaviside(clk - Kd5)*heaviside(KdA - not_q) + alpha4*heaviside(Kd6 - q)* heaviside(KdA - not_q)  - delta2*not_q	
	
	return np.array([da_dt, dnot_a_dt, dq_dt, dnot_q_dt])
	
#heaviside function
def heaviside(s): 
	return 0.5*(np.sign(s) + 1)
	
#gets sumed difderence of arrayData  
def getDif(indexes, arrayData):	
	arrLen = len(indexes)
	sum = 0
	for i, ind in enumerate(indexes):
		if i == arrLen - 1:
			break
		sum += arrayData[ind] - arrayData[indexes[i + 1]]
	return sum 
	
#gets standard deviation
def getSTD(indexes, arrayData, window):
	arrLen = len(arrayData)
	sum = 0
	for ind in indexes:
		minInd = max(0, ind - window)
		maxInd = min(arrLen, ind + window)
		sum += np.std(arrayData[minInd:maxInd])
	return sum
	
	
#elbow method
def elbow_method(subjects, k = 20, plot = False):   

	X = np.array(subjects)

	colors = ['b', 'g', 'r']
	markers = ['o', 'v', 's'] 
	 
	# k means for optimal k
	distortions = []
	K = range(1,k+1)
	for k in K:
		print(k)
		kmeanModel = KMeans(n_clusters=k).fit(X)
		kmeanModel.fit(X)
		distortions.append(kmeanModel.inertia_) 
	 
	# Plot the elbow
	plt.subplot(1, 2, 1) 
	ax = plt.gca()
	ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))	
	ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
	ax.yaxis.set_major_formatter(ticker.FuncFormatter( lambda x, y: "$" + '{:1.0f}'.format(x/1e8) + "x10^{8}$"	)) 	
	lines = plt.plot(K, distortions, 'o-', color='dodgerblue')
	plt.setp(lines[0], linewidth=1.5) 
	plt.xlabel(r"k" + "\n \n" + r"$\mathbf{a)}$")  
	plt.ylabel('Vsota kvadratnih napak')
	
	if plot:
		plt.show() 

#gap statistic method	
def gapStatistics(referenceSolver, subjects, number_ref = 10, max_clusters = 20, plot = False): 
	subjects = np.array(subjects)   
	sample_size = 250       
	
	gaps = []
	deviations = []
	references = []
	clusters_range = range(1, max_clusters + 1)

	for gap_clusters in clusters_range:
		print(gap_clusters) 
		reference_inertia = []	
		for index in range(number_ref):
			reference = np.array(referenceSolver.toolbox.population(sample_size)) 
			kmeanModel = KMeans(gap_clusters) 
			kmeanModel.fit(reference) 
			reference_inertia.append(kmeanModel.inertia_)    
		
		kmeanModel = KMeans(gap_clusters)    
		kmeanModel.fit(subjects)    
		log_ref_inertia = np.log(reference_inertia)	
		#calculate gap
		gap = np.mean(log_ref_inertia) - np.log(kmeanModel.inertia_) 
		sk = math.sqrt(1 + 1.0/number_ref)*np.std(log_ref_inertia)  
		gaps.append(gap) 
		deviations.append(sk)    
		
	# Plot the gaps 
	plt.subplot(1, 2, 2)	
	ax = plt.gca()
	ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))	 
	ax.xaxis.set_major_locator(ticker.MultipleLocator(2))	
	lines = plt.errorbar(clusters_range, gaps, yerr=sk, ecolor='dodgerblue', fmt='-', color='dodgerblue') 
	plt.setp(lines[0], linewidth=1.5) 
	plt.xlabel(r"k" + "\n \n" + r"$\mathbf{b)}$")     
	plt.ylabel('Vrzeli')
	
	if plot:
		plt.show()  

	#return optimal number of clusters
	for k in range(0, max_clusters - 1): 
		if gaps[k] >= gaps[k + 1] - deviations[k + 1]: 
			return k + 1     
		
	return max_clusters  	
 
#performs hierarchical clustering and plots dendogram 
def hierarchicalClustering(subjects):   
	X = np.array(subjects)
	# generate the linkage matrix
	Z = hierarchy.ward(X) 
	 
	matplotlib.rcParams['lines.linewidth'] = 1.5
	
	hierarchy.set_link_color_palette(['crimson', 'dodgerblue', 'seagreen', 'k'])   #
	
	# plot full dendrogram# 
	plt.figure(figsize=(15, 8))
	plt.title('Dendogram hierarhičnega razvrščanja')
	plt.xlabel('Vzorci')
	plt.ylabel('Wardova razdalja')  
	hierarchy.dendrogram(Z,leaf_rotation=90., leaf_font_size=0., no_labels=True, color_threshold=3000,  above_threshold_color='k') 
	plt.show()  

#calculates silhuetes and plots them	
def plot_silhouettes(X, y):
	
	LABEL_COLOR_MAP = {0 : 'crimson', 1 : 'dodgerblue', 2 : 'limegreen', 3 : 'yellow'} 
	cluster_labels = np.unique(y)
	n_clusters = cluster_labels.shape[0]
	silhouette_vals = metrics.silhouette_samples(X, y, metric='euclidean')
	y_ax_lower = 0
	y_ax_upper = 0
	yticks = []
	for i, c in enumerate(cluster_labels):
		c_silhouette_vals = silhouette_vals[y == c]
		c_silhouette_vals.sort()
		y_ax_upper += len(c_silhouette_vals)
		color = LABEL_COLOR_MAP[c] 
		plt.barh(
            range(y_ax_lower, y_ax_upper),
            c_silhouette_vals, 
            height=1.0,
            edgecolor='none',
            color=color,
			rasterized=True
        ) 
		yticks.append((y_ax_lower + y_ax_upper) / 2)
		y_ax_lower += len(c_silhouette_vals)

	plt.yticks(yticks, cluster_labels + 1) 
	plt.ylabel('Razred')
	plt.xlabel('Koeficient silhuete')  
	plt.show() 	

#read the batch of individuals in array from file 	
def readData(filename): 
	results_space = []
	results_cost = []	
	results_amp = []
	results_per = [] 

	#open file
	f = open(filename, "rb") 
	i = 0 
	while 1:
		try:
			#read next line from file 
			params = pickle.load(f) 
			if  params[2] > 0:  
				if params[1] == 0:
					print("zero cost")
				results_space.append(params[0])  
				results_cost.append(params[1])  
				results_amp.append(params[2])   
				results_per.append(params[3])  
				i = i + 1 				
		except (EOFError, pickle.UnpicklingError):
			break 
	
	f.close()  
	print(i)   
	return (np.array(results_space), np.array(results_cost), np.array(results_amp), np.array(results_per)) 

#performs PCA anylsis and plots the results
def plotPCA(data, project, labels):    
	pca = decomposition.PCA(n_components=3)  
	sc = preprocessing.StandardScaler() 
	data = sc.fit_transform(data)
	project = sc.transform(project) 
	pca.fit(data)     
	X = pca.transform(project)      	
	fig = plt.figure(1)
	plt.cla()
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134) 
	ax = plt.gca()
	ax.scatter(X[:, 0], X[:, 1],  X[:, 2], c=labels, edgecolor='k', rasterized=True)      
	ax.set_xlabel('GK 1')    
	ax.set_ylabel('GK 2') 
	ax.set_zlabel('GK 3')    	
	plt.show() 

#analyse given clusters	
def analyseClusters(data_space, data_cost, data_amp, data_per, n_clusters, labels, solver):  
  
	centers = []
	centers_top = []
	clusters = [] 
	clusters_top = [] 
	clusters_labels = []  
	clusters_labels_top = []   
	
	titles = [r"$\mathbf{a)}$", r"$\mathbf{b)}$", r"$\mathbf{c)}$", r"$\mathbf{d)}$"] 
	
	for i, cluster_label in enumerate(range(n_clusters)):  
		ind = np.where(labels == cluster_label)[0]
		clusterSize = np.shape(ind)[0]
		clusterPoints = data_space[ind] 
		clusters.extend(clusterPoints) 
		cluster_labels = labels[ind]
		clusters_labels.extend(cluster_labels)     
		
		costs = data_cost[ind]   
		amps = data_amp[ind]  
		pers = data_per[ind]    
		cluster_center = np.mean(clusterPoints, axis = 0)     
		cluster_inertia = np.sum(cdist(clusterPoints, [cluster_center], 'euclidean'))*(1.0/clusterSize)  
		#select top 10% 
		percentage = 0.1 
		costs_index = np.array(sorted(range(len(costs)), key=lambda k: costs[k]))
		index_top = costs_index[0:int(costs_index.shape[0]*percentage)]
		clusterPoints_top = clusterPoints[index_top] 
		top_center = np.mean(clusterPoints_top, axis = 0)
		
		print("Cluster index: ", cluster_label)     		
		print("Cluster inertia: ", cluster_inertia)  
		print("Cluster size: ", clusterSize)     		
		print("Avg. Cost: ", np.mean(costs))  
		print("Dev. Cost: ", np.std(costs)) 
		print("Avg. Amp: ", np.mean(amps))    
		print("Dev. Amp: ", np.std(amps))   		
		print("Avg. Per: ", np.mean(pers))  
		print("Dev. Per: ", np.std(pers)) 		
		print("Center: ", cluster_center)    
		print("Center cost: ", -1*solver.evalIndividual(cluster_center)[0]) 
		print("Top Center: ", top_center)      		
		print("Top Center Cost: ", -1*solver.evalIndividual(top_center)[0])   
		
		plt.subplot(2,2, i+1)
		solver.plotModel(top_center, titles[i])    

		clusters_top.extend(clusterPoints_top)        
		clusters_labels_top.extend(cluster_labels[index_top])          
	plt.show()
	return (np.array(clusters_top), np.array(clusters_labels_top))   

#scatter plot 3D
def plot3D(data, labels, i, j, k, ax):  
	ax.scatter(data[:, i], data[:, j], data[:, k], c=labels, edgecolor='k',  rasterized=True)         	   

#scatter plot 1D 	
def plot1D(data, cost, i):	
	x = data[:, i]
	z = cost 	
	index = np.array(sorted(range(len(x)), key=lambda k: x[k]))
	x = x[index]
	z = z[index]  
	plt.plot(x,z,'.',rasterized=True)     	 
		          
if __name__ == "__main__": 
	
	#search the parameter space of represilator model
	geneticSolverRepres = geneticSolver(repressilatorModel, np.array([0 , 100, 0, 0, 0, 0]), np.array(["transcription", "transcription", "hill", "translation", "rna_degradation", "protein_degradation", "Kd"]), minCost = 6*1e4, indpb = 0.5) 
	for i in range(0, 50):  
		print("represilator space search iteration %d" % (i+1))      
		geneticSolverRepres.run()    

	#search the parameter space of D flip flop model 
	geneticSolverFlipFlop = geneticSolver(flipFlopModel, np.array([0.0 , 0.0, 100.0, 0.0]), np.array(["protein_production", "protein_production", "Kd", "Kd", "Kd", "Kd", "protein_degradation", "protein_production", "protein_production", "Kd", "Kd", "Kd", "protein_degradation"]), minCost = -540000, dt = 0.01) 
	for i in range(0, 50):  
		print("flip flopa space search iteration %d" % (i+1))   
		geneticSolverFlipFlop.run()         
		
	#color codes
	LABEL_COLOR_MAP = {0 : 'crimson', 1 : 'dodgerblue', 2 : 'limegreen', 3 : 'yellow'}

	############ REPRESILATOR ANALYSIS ############ 
	
	(results_rep_space, results_rep_cost, results_rep_amp, results_rep_per) = readData("results_rep_genetic")  
	solver = geneticSolver(repressilatorModel, np.array([0 , 100, 0, 0, 0, 0]), np.array(["transcription", "transcription", "hill", "translation", "rna_degradation", "protein_degradation", "Kd"]))	
	results_rep_cost = -results_rep_cost
	
	#get optimal number of clusters
	elbow_method(results_rep_space)  
	optimalK = gapStatistics(solver, results_rep_space) 
	print(optimalK)  
	#performs kmenas clustering 
	n_clusters = 4   
	kmeanModel = KMeans(n_clusters).fit(results_rep_space)       
	labels = kmeanModel.labels_    

	#analyse clusters and plots silhuete and PCA 
	(clusters_top, clusters_labels_top) = analyseClusters(results_rep_space, results_rep_cost, results_rep_amp, results_rep_per, n_clusters, labels, solver) 
	label_colors = np.array([LABEL_COLOR_MAP[l] for l in clusters_labels_top])
	ind = np.random.choice(range(results_rep_space.shape[0]), 1000)     
	plot_silhouettes(results_rep_space[ind], labels[ind])      	
	plotPCA(results_rep_space, clusters_top, label_colors)    

	#plots 3d scatter plot 
	scatterLabels = [(r"$n$" + "\n", r"$\delta_{m}$", r"$\delta_{p}$"), (r"$n$", r"$\delta_{p}$", r"$K_{d}$"), (r"$\delta_{m}$", r"$\delta_{p}$",  r"$K_{d}$")]  
	titles = [r"$\mathbf{a)}$", r"$\mathbf{b)}$", r"$\mathbf{c)}$"]  
	axes3d = [(2,4,5), (2,5,6), (4,5,6)]    
	fig = plt.figure()
	for ind, plot3d in enumerate(axes3d):
		ax = fig.add_subplot(1, 3, ind + 1, projection='3d')	
		plot3D(clusters_top, label_colors, plot3d[0], plot3d[1], plot3d[2], ax) 	
		ax.set_title(titles[ind]) 
		ax.set_xlabel(scatterLabels[ind][0])  
		ax.set_ylabel(scatterLabels[ind][1])
		ax.set_zlabel(scatterLabels[ind][2])   		 		
	plt.show()  

	#plots 1D scatter plot 
	axes = [0,1,3,4,5,6]
	xlabels = [r"$\alpha$" + "\n" + r"$\mathbf{a)}$", r"$\alpha_{0}$" + "\n" + r"$\mathbf{b)}$",r"$\beta$" + "\n" + r"$\mathbf{c)}$",r"$\delta_{m}$" + "\n" + r"$\mathbf{d)}$",r"$\delta_{p}$" + "\n" + r"$\mathbf{e)}$",r"$K_{d}$" + "\n" + r"$\mathbf{f)}$"]    
	ind = 1   
	fig, ax = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=True, figsize=(6, 6))
	for i in axes:    
		plt.subplot(3,2,ind) 
		plt.xlabel(xlabels[ind - 1])   
		ax = plt.gca()
		ax.invert_yaxis()
		ind += 1  
		plot1D(results_rep_space, results_rep_cost, i) 
	fig.text(0.004, 0.5, 'Optimizacijski kriterij', va='center', rotation='vertical') 	
	plt.show()      		

	#plot heatmaps 
	xlabels = [r"$\alpha$" + "\n" + r"$\mathbf{a)}$", r"$\alpha$" + "\n" + r"$\mathbf{b)}$",r"$\beta$" + "\n" + r"$\mathbf{c)}$",r"$\beta$" + "\n" + r"$\mathbf{d)}$"] 
	ylabels = [r"$\beta$", r"$\delta_{p}$", r"$\delta_{m}$", r"$\delta_{p}$"]     
	ind = np.random.choice(range(results_rep_space.shape[0]), 10000)  	
	heatAxes = [(0,3),(0,5), (3,4), (3,5)]    
	k = 1
	for heatAxis in heatAxes:  
		i = heatAxis[0] 
		j = heatAxis[1]   
		x = results_rep_space[:, i]  
		y = results_rep_space[:, j]    
		z = results_rep_cost     
		x = x[ind] 
		y = y[ind] 
		z = z[ind]    
		
		#interpolates the data on a grid 
		f = interp2d(x,y,z, kind='linear')            
		xi = np.linspace(min(x), max(x), 150)    	
		yi = np.linspace(min(y), max(y), 150)   
		ZI = f(xi,yi)
		plt.subplot(2,2,k)   
		plt.xlabel(xlabels[k - 1]) 
		plt.ylabel(ylabels[k - 1])    
		
		XI, YI = np.meshgrid(xi, yi)     
		plt.pcolor(XI, YI, ZI, cmap=cm.jet_r, vmax=0, rasterized=True)  
		plt.xlim(min(xi), max(xi))
		plt.ylim(min(yi), max(yi))   			
		plt.colorbar()  
		
		k += 1 
	
	plt.show()  


	############ FLIP FLOP ANALYSIS ############  	
	(results_flipflop_space, results_flipflop_cost, results_flipflop_amp, results_flipflop_per) = readData("results_flipflop_genetic") 
	results_flipflop_cost = -results_flipflop_cost   	
	solver = geneticSolver(flipFlopModel, np.array([0.0 , 0.0, 100.0, 0.0]), np.array(["protein_production", "protein_production", "Kd", "Kd", "Kd", "Kd", "protein_degradation", "protein_production", "protein_production", "Kd", "Kd", "Kd", "protein_degradation"]), dt = 0.01) 	
	
	#get optimal number of clusters 
	elbow_method(results_flipflop_space) 
	optimalK = gapStatistics(solver, results_flipflop_space)  
	print(optimalK)
	#performs kmenas clustering  	
	n_clusters = 4
	kmeanModel = KMeans(n_clusters).fit(results_flipflop_space)        
	labels = kmeanModel.labels_  
	
	#analyse clusters and plots silhuete and PCA 
	(clusters_top, clusters_labels_top) = analyseClusters(results_flipflop_space, results_flipflop_cost, results_flipflop_amp, results_flipflop_per, n_clusters, labels, solver) 
	label_colors = np.array([LABEL_COLOR_MAP[l] for l in clusters_labels_top])  
	ind = np.random.choice(range(results_flipflop_space.shape[0]), 1000)  	
	plot_silhouettes(results_flipflop_space[ind], labels[ind])     	
	plotPCA(results_flipflop_space, clusters_top, label_colors)       	
		
	#plots 3d scatter plot 
	scatterLabels = [(r"$\alpha_{1}$", r"$\alpha_{2}$", r"$\alpha_{3}$"), (r"$\alpha_{1}$", r"$Kd_{7}$", r"$\delta_{1}$"), (r"$\alpha_{2}$", r"$\alpha_{3}$",  r"$Kd_{5}$")]    
	titles = [r"$\mathbf{a)}$", r"$\mathbf{b)}$", r"$\mathbf{c)}$"]      		 
	axes3d = [(0,1,7), (0,5,6), (0,7,10)] 
	fig = plt.figure()
	for ind, plot3d in enumerate(axes3d):
		ax = fig.add_subplot(1, 3, ind + 1, projection='3d')	
		plot3D(clusters_top, label_colors, plot3d[0], plot3d[1], plot3d[2], ax) 
		ax.set_title(titles[ind])  
		ax.set_xlabel(scatterLabels[ind][0])  
		ax.set_ylabel(scatterLabels[ind][1])
		ax.set_zlabel(scatterLabels[ind][2])   		 		
	plt.show()  	
		
	#plots 1D scatter plot 
	axes = [7, 8, 10, 11, 5, 12]
	xlabels = [r"$\alpha_{3}$" + "\n" + r"$\mathbf{a)}$",r"$\alpha_{4}$" + "\n" + r"$\mathbf{b)}$",r"$Kd_{5}$" + "\n" + r"$\mathbf{c)}$",r"$Kd_{6}$" + "\n" + r"$\mathbf{d)}$", r"$Kd_{7}$" + "\n" + r"$\mathbf{e)}$", r"$\delta_{2}$" + "\n" + r"$\mathbf{f)}$"]    	
	ind = 1
	fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(6, 6))   
	for i in axes:  
		plt.subplot(3,2,ind)   
		plt.xlabel(xlabels[ind - 1])	 	
		ax = plt.gca()	
		ax.invert_yaxis()	
		ind += 1
		plot1D(results_flipflop_space, results_flipflop_cost, i) 
		ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, y: "$" + locale.format('%1.0f',(x)/1e5) + "x10^{5}$" ))
	fig.text(0.004, 0.5, 'Optimizacijski kriterij', va='center', rotation='vertical')  	
	plt.show()     
	
	#plot heatmaps 
	xlabels = [r"$Kd_{7}$" + "\n" + r"$\mathbf{a)}$", r"$Kd_{7}$" + "\n" + r"$\mathbf{b)}$",r"$Kd_{7}$" + "\n" + r"$\mathbf{c)}$",r"$Kd_{7}$" + "\n" + r"$\mathbf{d)}$"] 
	ylabels = [r"$\delta_{1}$", r"$\alpha_{3}$", r"$Kd_{5}$", r"$\delta_{2}$"] 	
	heatAxes = [(5,6), (5,7), (5,10), (5,12)]       
	ind = np.random.choice(range(results_flipflop_space.shape[0]), 10000)      
	k = 1
	for heatAxis in heatAxes:   
		i = heatAxis[0] 
		j = heatAxis[1]     
		
		x = results_flipflop_space[:, i]
		y = results_flipflop_space[:, j] 
		z = results_flipflop_cost     
		z = z - np.max(z)   

		x = x[ind]			
		y = y[ind]   
		z = z[ind]   

		xi = np.linspace(np.min(x), np.max(x), 150)     	
		yi = np.linspace(np.min(y), np.max(y), 150)		
		
		x = np.append(x, [np.min(x)]*np.shape(np.unique(yi))[0])
		y = np.append(y, np.unique(yi))
		z = np.append(z, [0]*np.shape(np.unique(yi))[0])
	
		#interpolates the data on the grid
		f = interp2d(x,y,z, kind='linear')                
		ZI = f(xi,yi)     
		XI, YI = np.meshgrid(xi, yi)      
		plt.subplot(2,2,k) 
		plt.pcolor(XI, YI, ZI, cmap=cm.jet_r, vmax=np.max(z), vmin=np.min(z), rasterized=True)  		
		plt.xlim(min(xi), max(xi))
		plt.ylim(min(yi), max(yi))  		
		plt.colorbar(format=ticker.FuncFormatter(lambda x, y: "$" + locale.format('%1.0f',(x-np.min(z))/1e5) + "x10^{5}$" )) 
		plt.xlabel(xlabels[k - 1]) 
		plt.ylabel(ylabels[k - 1])   		
		
		k += 1	
	plt.show()     	
		
