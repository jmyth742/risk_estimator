import numpy as np
import time
from datetime import datetime
import random
import math
from matplotlib import pyplot as plt
from IPython.display import clear_output

PI = 3.1415926
e = 2.71828


age = 27
#0 signifies age 0-9, 1 signifies 10-19 etc
age_factor = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#'heart_conditions','asthma','diabetes','cancer','neurological','liver','kidney','fat','pregnant','immune_issues'
#no health issues = 0.1
#worst case scenario = 0.5 i.e heart disease. 
health_issues = [0.1,0.2,0.5]


#scale of levels of hygeniene wrt washing hands etc -- 1 being very and 5 not so much
personal_hygiene = [0.1,0.2,0.3,0.4,0.5]
population_destiny = 100


#time since encounter with person in days.
time_since_encounter = [0,1,2,3,4,5,6,7]
no_encounters = 25


#increments of 30 minutes
time_spent_outdoors = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
encounter_age = 100
encounter_risk_factor = 100


# location_of_user
# climate 
# encounter age, encounters risk factor,
list_of_encounters = np.zeros((no_encounters,1))
#print(list_of_encounters)

for i in range(25):
    rand_age = np.random.randint(0,99)
    #list_of_encounters[i,0] = rand_age
    list_of_encounters[i,0] = np.random.uniform(0.0, 0.99)


#print(list_of_encounters)


mode_of_transport = ['car','bus','train','bike','walking']

#gov't policies
#high lockdown = 1, no real lockdown = 6
lockdown_level = [1,2,3,4,5,6]



##
# this idea of infection probability encourter was taken from here in this paper.
# 
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6265857/
#
#


# effecctive radius of virus in meters
r_o = 2.0
# background transmission
B_o = 1.0
 # this is the mean but can also be 3 and 6.4 days  -- can vary
mean_incubation = 5.84
#standard deviation
s_d_incubation = 2.93
#mean infectious period 
mean_infectious_period = 9.0
#standard deviation
s_d_infectious = 3.0
#super speader proportion
lmbda = 0.1
#exponent in transmission rate
alpha = 2

def probability_infected(distance):
    if distance > r_o:
        return 0.0
    else:
        return B_o*(1-distance/r_o)**alpha


def random_timestamp():
    dt = datetime(2020, 3, 1) + random.random() * datetime.timedelta(days=1)
    return dt

series = np.zeros([200,2])


#
# here we want to just generate some random data for testing the alogrithm
#  
# #

var = random.sample(range(1,500),200)
data = sorted(var)
print(len(data))


#
# populate infectious profile with random proximity of encounters
# #

def sim_data():
    steps = 200
    for i in range(steps):
        series[i,0] = data[i]
        series[i,1] = probability_infected(random.uniform(0,1.5))
    #print(series)
    return series.shape

test = sim_data()


#
# f(x) function for the monte carlo simulation
# #

def f_of_x(x):
    """
    This is the main function we want to integrate over.
    Args:
    - x (float) : input to function; must be in radians
    Return:
    - output of function f(x) (float)
    """
    return (e**(-1*x))/(1+(x-1)**2)

#
# crude montecarlo simulation 
# #


def crude_monte_carlo(num_samples=200):
    """
    This function performs the Crude Monte Carlo for our
    specific function f(x) on the range x=0 to x=5.
    Notice that this bound is sufficient because f(x)
    approaches 0 at around PI.
    Args:
    - num_samples (float) : number of samples
    Return:
    - Crude Monte Carlo estimation (float)
    
    """
    lower_bound = np.min(series[:,1])
    print('lower_bound is ',lower_bound)
    upper_bound = np.max(series[:,1])
    print('upper_bound is ', upper_bound)
    
    sum_of_samples = 0
    for i in range(num_samples):
        x = series[i,1]
        
        sum_of_samples += f_of_x(x)
    
    return (upper_bound - lower_bound) * float(sum_of_samples/num_samples)



#
# crude monte carlo variance calculator
# #



def get_crude_MC_variance(num_samples):
    """
    This function returns the variance fo the Crude Monte Carlo.
    Note that the inputed number of samples does not neccissarily
    need to correspond to number of samples used in the Monte
    Carlo Simulation.
    Args:
    - num_samples (int)
    Return:
    - Variance for Crude Monte Carlo approximation of f(x) (float)
    """
    int_max = 5 # this is the max of our integration range
    
    # get the average of squares
    running_total = 0
    for i in range(num_samples):
        x = series[i,1]
        running_total += f_of_x(x)**2
    sum_of_sqs = running_total*int_max / num_samples
    
    # get square of average
    running_total = 0
    for i in range(num_samples):
        x = series[i,1]
        running_total = f_of_x(x)
    sq_ave = (int_max*running_total/num_samples)**2
    
    return sum_of_sqs - sq_ave





crude_estimation   = crude_monte_carlo(200)
variance           = get_crude_MC_variance(200)
error              = math.sqrt(variance / 200)
print('prediction is ', crude_estimation)
print('monte carlo variance ', variance)
print('error in approximation ', error)

#
# lets now plot the information
# #




# plot the function
xs = [float(i/50) for i in range(int(50*PI*2))]
ys = [f_of_x(series[int(x),1]) for x in xs]

plt.plot(xs,ys)
plt.title("f(x)")
plt.show()


# this is the template of our weight function g(x)
def g_of_x(x, A, lamda):
    e = 2.71828
    return A*math.pow(e, -1*lamda*x)


xs = [float(i/50) for i in range(int(50*PI))]
fs = [f_of_x(series[int(x),1]) for x in xs]
gs = [g_of_x(series[int(x),1], A=1.4, lamda=1.4) for x in xs]
plt.plot(xs, fs)
plt.plot(xs, gs)
plt.title("f(x) and g(x)");
plt.show()




def inverse_G_of_r(r, lamda):
    return (-1 * math.log(float(r)))/lamda




def get_IS_variance(lamda, num_samples):
    """
    This function calculates the variance if a Monte Carlo
    using importance sampling.
    Args:
    - lamda (float) : lamdba value of g(x) being tested
    Return: 
    - Variance
    """
    A = lamda
    int_max = 5
    
    # get sum of squares
    running_total = 0
    for i in range(num_samples):
        x = series[int(i),1]
        running_total += (f_of_x(x)/g_of_x(x, A, lamda))**2
    
    sum_of_sqs = running_total / num_samples
    
    # get squared average
    running_total = 0
    for i in range(num_samples):
        x = series[int(i),1]
        running_total += f_of_x(x)/g_of_x(x, A, lamda)
    sq_ave = (running_total/num_samples)**2
    
    
    return sum_of_sqs - sq_ave

# get variance as a function of lambda by testing many
# different lambdas

test_lamdas = [i*0.05 for i in range(1, 61)]
variances = []

for i, lamda in enumerate(test_lamdas):
    #print(f"lambda {i+1}/{len(test_lamdas)}: {lamda}")
    print("lambda ",(i+1/len(test_lamdas)))
    A = lamda
    variances.append(get_IS_variance(lamda, 200))
    clear_output(wait=True)
    
optimal_lamda = test_lamdas[np.argmin(np.asarray(variances))]
IS_variance = variances[np.argmin(np.asarray(variances))]

print("Optimal Lambda: ", optimal_lamda)
print("Optimal Variance: ", IS_variance)
print((IS_variance/10000)**0.5)

plt.plot(test_lamdas[5:40], variances[5:40])
plt.title("Variance of MC at Different Lambda Values");
plt.show()



def importance_sampling_MC(lamda, num_samples):
    A = lamda
    
    running_total = 0
    for i in range(num_samples):
        r = series[int(i),1]
        running_total += f_of_x(inverse_G_of_r(r, lamda=lamda))/g_of_x(inverse_G_of_r(r, lamda=lamda), A, lamda)
    approximation = float(running_total/num_samples)
    return approximation



# run simulation
num_samples = 200
approx = importance_sampling_MC(optimal_lamda, num_samples)
variance = get_IS_variance(optimal_lamda, num_samples)
error = (variance/num_samples)**0.5

# display results
print("Importance Sampling Approximation:" ,approx)
print("Variance:", variance)
print("Error:" ,error)

