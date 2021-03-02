from sympy.interactive import printing
printing.init_printing(use_latex=True)
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


from sympy import Eq, solve_linear_system, Matrix
from numpy import linalg
import numpy as np
import sympy as sp
from sympy import *

x, mu, sigma, w, u, v, theta, r, l, k = sp.symbols('x mu sigma w u v theta r lambda, k')

### Create the states


A1_init = 10 ## C[0]
A2_init = 10 ## C[1]
E_init = 0 ## C[2]
S_init = 0 ## C[3]
A1_prot_init = 0 ## C[4]
A1_inact_init = 0 ## C[5]

##for quick lookup to avoid errors
A1_idx = 0
A2_idx = 1
E_idx = 2
S_idx = 3
A1_prot_idx = 4
A1_inact_idx = 5
dead_idx = -1

## The rates
k1 = 1
k2 = 1
kd = 1
k4 = 1
k3 = 100
k5 = 100

C_states = [A1_init, A2_init, E_init, S_init, A1_prot_init, A1_inact_init]


### function to calculate propensities



reaction_table = [
    {'reactant': [A1_idx], 'product': [E_idx, A1_idx], 'rate': k1},  ## REACTION 1 >>> reactants A1 , products E and A1
    {'reactant': [A2_idx ], 'product': [ S_idx , A2_idx], 'rate': k2},  ## REACTION 2 >>> reactants A2 , products S and A2
    {'reactant': [E_idx, A1_idx], 'product': [E_idx , A1_prot_idx], 'rate': k3},  ## REACTION 3 >>> reactants E, A1 , products E and A1_prot
    {'reactant': [A1_prot_idx], 'product': [E_idx, A1_prot_idx], 'rate': k4},  ## REACTION 4 >>> reactants A1_prot , products E and A1_prot
    {'reactant': [S_idx, A1_idx], 'product': [S_idx, A1_inact_idx], 'rate': k5},  ## REACTION 5 >>> reactants S A1 , products S A1_inact
    {'reactant': [S_idx], 'product': [dead_idx], 'rate': kd},  ## REACTION 6 >>> reactants S , products nothing
    {'reactant': [E_idx], 'product': [dead_idx], 'rate': kd},  ## REACTION 7 >>> reactants S , products nothing
]

### Create each of the reactions
#### NOTE this does not work for A + A >> A2 reactions yet, code must be altered
#### RETURNS: A vector of propensities, and the total propensities
def calc_propensities_to_categ_parameters(reaction_table, C_states): ##input reaction table and states

    individual_propensities = []
    categorical_dist_parameters = []

    for i in range(len(reaction_table)):
        ## calculate the propensity for u_1
        prop = 1
        for j in range(len(reaction_table[i]['reactant'])):
            state_index = reaction_table[i]['reactant'][j] ## looking up the reactants for reaction i
            prop = prop * C_states[state_index] ## multiplying each reactant population by propensity

        prop = prop * reaction_table[i]['rate']
#         print("==> Total Propensity for reaction: ", i , " is " , prop)
        individual_propensities.append(prop)

    total_propensity = sum(individual_propensities)
    for i in range(len(individual_propensities)):
        ## sum the propoensities and divide it in each of the individual propensities and append to categorical_dist_parameters
        categorical_dist_parameters.append(individual_propensities[i] / total_propensity)



    return categorical_dist_parameters, total_propensity

parameters, tot_prop = calc_propensities_to_categ_parameters(reaction_table, C_states)
parameters


def sample_from_categorical(parameters, u):
    payload = 0
    count = 0
    while (u > payload):
        payload += parameters[count]
        count += 1
    return count-1

sample_from_categorical(parameters, 0.01)

lambda_parameter = 1

def exponential_holding_time(lambda_parameter, uniformly_sampled):
    l_s = lambda_parameter
    inverse_cdf = - 1 / l_s * log(1 - u)
    holding_time = inverse_cdf.subs(u, uniformly_sampled)

    return holding_time

exponential_holding_time(lambda_parameter, 0.1)


def reaction_happens(reaction_table, reaction_idx, C_states):
    reactant_list = reaction_table[reaction_idx]['reactant']
#     print("REACTANT LIST: ", reactant_list)
    product_list = reaction_table[reaction_idx]['product']
    new_c_states = C_states
    ## take away reactants
    for i in range(len(reactant_list)):
        state_idx = reactant_list[i]
        new_c_states[state_idx] = C_states[state_idx] - 1

    ## add products
    for i in range(len(product_list)):
        state_idx = product_list[i]
        if(state_idx >= 0):
            new_c_states[state_idx] = C_states[state_idx] + 1

    return new_c_states

C_states = [10, 10, 0.0, 0.0, 0.0, 0.0, 0.0]

reaction_happens(reaction_table, 0, C_states)

C_states = [A1_init, A2_init, E_init, S_init, A1_prot_init, A1_inact_init]

print(C_states)

C_states_history = []
A1_hist = []
A2_hist = []
E_hist = []

holding_times_history = []
total_time = 0

C_states_history.append(list(C_states))
holding_times_history.append(0)

for i in range(1000):
    ### putting the code together

    ## 1. Sample the reaction that will take place
    parameters, tot_prop = calc_propensities_to_categ_parameters(reaction_table, C_states)
#     print(C_states)
#     print(parameters)

    u = np.random.uniform(0,1,1)[0]
#     print("\t sampled u: => ", u)
    reaction_idx = sample_from_categorical(parameters, u)
#     print("\t REACTION: => ", reaction_idx)
    ## update C_states based on the reaction chosen

    C_states = reaction_happens(reaction_table, reaction_idx, C_states)
    C_states_history.append(list(C_states))

    h_time = exponential_holding_time(tot_prop, np.random.uniform(0,1,1)[0])
    total_time += h_time
    holding_times_history.append(total_time)
# print(C_states_history)

a = np.array(C_states_history)
A1_hist = a[:,A1_idx]
A2_hist = a[:,A2_idx]
S_hist = a[:, S_idx]
E_hist = a[:, E_idx]
A1_prot_history = a[:,A1_prot_idx]

# print((A1_prot_history))
# print((holding_times_history))
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') ## just code for enlarging the graph

plt.step(holding_times_history , A1_prot_history, label="A1_prot")
plt.step(holding_times_history, A1_hist, label="A1")
plt.step(holding_times_history , A2_hist, label="A2")
plt.step(holding_times_history , S_hist, label="S")
plt.step(holding_times_history , E_hist, label="E")

plt.title('History of the A1 and A2 and S and E')
plt.legend()
plt.ylabel('Amount of A1')
plt.xlabel('Elapsed Time');

print(A1_prot_history[-1])


#### PUTTING IT ALL TOGETHER

from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

## creating initial conditions
initial_conditions = np.zeros(100)
initial_conditions[0] = 1

## running the process
time_elapsed = 0
current_number = 0
time_elapsed_list = []
member_history = []

for i in range(1000):

    ## multiplying the initial conditions with the Transition matrix to simulate the next step
    initial_conditions = np.matmul(initial_conditions, PI)
    # sampling froma categorical distribution to represent the current number of creatures
    current_number = sample_from_categorical(initial_conditions, np.random.uniform(0,1,1)[0])
    # holding times
    Ht = exponential_holding_time(current_number)

    time_elapsed += Ht
    time_elapsed_list.append(time_elapsed)
    member_history.append(current_number)

plt.step(time_elapsed_list , member_history)
plt.title('Birth-Death Poisson Process')
plt.ylabel('Amount of RNA')
plt.xlabel('Elapsed Time');
