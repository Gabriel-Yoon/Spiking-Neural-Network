import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time as tm


"""
<Basic parameters>
"""

T       = 20 # total time to simulate (in msec)
dt      = 0.0125 # simulation timestep
tnow    = 0     # current time for the simulation (processing time)
timeflag = 0    # for while loop execution
timing    = int(T/dt) #number of timesteps in the total simulation time
inpt    = 1.0 #Neuron input voltage (in V)
neuron_input = np.full((time),inpt)

"""
-----------------------------------------------------------------------------------------------------------------------
START : RECORD Page
"""
# I'm thinking of appending the export spike information to the record paper
# technique : self.spk_rcd = np.append(spk_rcd, [[self.label],[tnow]], axis=1)



"""
END : RECORD Page
-----------------------------------------------------------------------------------------------------------------------
"""

"""
-----------------------------------------------------------------------------------------------------------------------
START : LIF neuron model
"""

class LIFNeuron():
    def __init__(self):
        # Simulation config (may not all be needed!!)
        self.dt = 0.125  # simulation time step

        # LIF Properties
        self.Vm     = 0       # Neuron potential (mV) over time
        self.time   = 0       # Time duration for the neuron

        self.clk    = 0        # neuron clk that records the latest spike timing
        self.type   = 'Leaky Integrate and Fire'
        self.t_ref  = 0.05    # refractory period (ms)
        self.ref_flag = int(self.t_ref/dt)  # numver of refractory times in the sim timestep
        self.V_th    = 0.75  #spike threshold voltage
        self.V_ref  = 0     #refractory time voltage
        self.V_spike = 1    # spike delta (V)
        self.label = 0      # the label of the neuron
        self.pot_rcd = np.array([[0],[0]])
        self.spk_rcd = np.array([[0],[0]])   # Spike time record paper in number-time

    def printParameters(self):
        print ('latest spike timing : ', self.clk)
        print ('Membrane potential : ', self.Vm)
    
def spike_generator(self, ext_spk_pot, tnow):
        if self.clk == 0 and self.spk_bool == 0 and self.Vm + ext_spk_pot >=0: # pristine state, accruing the potential from the initial state
            if self.Vm + ext_spk_pot < self.V_th:   # if they are not over the threshold
                self.Vm = self.Vm + ext_spk_pot     # just stack them!
                self.spk_rcd = np.append(self.spk_rcd, [[tnow],[0]], axis = 1)
            elif self.Vm + ext_spk_pot >= self.V_th: # time to spike!
                self.spk_rcd = np.append(self.spk_rcd, [[tnow],[1]], axis = 1)
                self.clk = tnow
                self.Vm = self.V_ref
                self.spk_bool = 1
        else:   # spike has generated at least once from the previous activities -> most of them entering this phase
            if tnow > self.clk + self.t_ref:                     # not in the refractory period
                if self.Vm + ext_spk_pot < self.V_th:        # not over the threshold
                    self.Vm = self.Vm + ext_spk_pot             # stack the spike
                    self.spk_rcd = np.append(self.spk_rcd, [[tnow],[0]], axis = 1)
                elif self.Vm + ext_spk_pot >= self.V_th and tnow > self.clk + self.t_ref:  # time to spike!
                    self.spk_rcd = np.append(self.spk_rcd, [[tnow],[1]], axis = 1)
                    self.Vm = self.V_ref
                    self.clk = tnow
            else:
                self.spk_rcd = np.append(self.spk_rcd, [[tnow],[0]], axis = 1)

        self.pot_rcd = np.append(self.pot_rcd, [[tnow],[self.Vm]], axis = 1)

    def random_walk(self):
        random_number = random.randint(0,1) - 0.5
        self.Vm += random_number *0.2
        
    def leak(self):
        self.Vm -= 0.1

test_neuron = LIFNeuron()
input_spike = np.array([0.75])

for i in range(timing):
    rng = 0.0001*random.randint(-600,1000)
    input_spike = np.append(input_spike, rng)

while tnow < T:
    if tnow == 0:
        print("Simulation Executed")

    test_neuron.spike_generator(input_spike[timeflag], tnow)

    tnow += dt
    timeflag += 1
    if tnow >= T:
        print("Simulation Exit")

"""
END : LIF Neuron model
-----------------------------------------------------------------------------------------------------------------------
"""

"""
START : Create neural network
-----------------------------------------------------------------------------------------------------------------------
"""

num_layers  = 2
num_neurons = 25

def create_neurons(num_layers, num_neurons):
    neurons = []
    for layer in range(num_layers):
        neuron_layer=[]
        for count in range(num_neurons):
            neuron_layer.append(LIFNeuron())
        neurons.append(neuron_layer)
    return neurons

neurons = create_neurons(num_layers, num_neurons) # Created neural network with num_layers and num_neurons


"""
END : Create neural network
-----------------------------------------------------------------------------------------------------------------------
"""



"""
-----------------------------------------------------------------------------------------------------------------------
START : GRAPH_RESULTS
Utility functions
These functions are used to graph the results of spikes, membrane potential, etc for neurons in the simulation.
"""
def plot_neuron_behaviour(time, data, neuron_type, neuron_id, y_title):
    #print ('Drawing graph with time.shape={}, data.shape={}'.format(time.shape, data.shape))
    plt.figure(figsize=(20,10))
    plt.plot(time,data)
    plt.title('{} at {}'.format(neuron_type, neuron_id))
    plt.ylabel(y_title)
    plt.xlabel('Time (msec)')
    # Graph to the data with some headroom...
    y_min = 0
    y_max = max(data)*1.2
    if y_max == 0:
        y_max = 1
    plt.ylim([y_min,y_max]) 
    plt.show()

def plot_membrane_potential(time, Vm, neuron_type, neuron_id = 0):
    plot_neuron_behaviour(time, Vm, neuron_type, neuron_id, y_title = 'Membrane potential (V)')

def plot_spikes(time, Vm, neuron_type, neuron_id = 0):
    plot_neuron_behaviour(time, Vm, neuron_type, neuron_id, y_title = 'Spike (V)')

"""
END : GRAPH_RESULTS
-----------------------------------------------------------------------------------------------------------------------
"""

stimulus_len = len(neuron_input) # Length of the stiumulus is analogous to that of the neuron_input
layer = 0
for neuron in range(num_neurons):
    offset = random.randint(0,100)  #Simulates stimulus starting at different times
    stimulus = np.zeros_like(neuron_input)
    stimulus[offset:stimulus_len] = neuron_input[0:stimulus_len - offset]
    neurons[layer][neuron].spike_generator(stimulus)

plot_membrane_potential(test_neuron.pot_rcd[0][:], test_neuron.pot_rcd[1][:], 'Membrane Potential for neuron {}'.format(test_neuron.label), neuron_id = "visible layer")
plot_spikes(test_neuron.spk_rcd[0][:], test_neuron.spk_rcd[1][:], 'Output spikes for neuron {}'.format(test_neuron.label), neuron_id = "visible layer")






"""
-----------------------------------------------------------------------------------------------------------------------
START : MNIST IMAGE_UTILS
Render the image in each of it's retinal 'zones'
This will be the basis of what each retinal unit views as we progress.
Not that the pixels that will have the strongest stimuli are white (as they are closer to 1), areas of least stimuli are black (value close to 0).
"""
from mnist import MNIST
mndata = MNIST('./mnist')
images, labels = mndata.load_training()


def get_next_image(index=0, pick_random = False, display=True):
    if pick_random:
        index = random.randint(0, len(images)-1)
    image = images[index]
    label = labels[index]
    if display:
        print('Label: {}'.format(label))
        print(mndata.display(image))
    image = np.asarray(image).reshape((28,28))
    image_norm = (image * 255.0/image.max()) / 255.
    return image_norm, label


def graph_retinal_image(image, stride):
    fig = plt.figure()

    len_x, len_y = image.shape
    x_max = int(len_x/stride[0])
    y_max = int(len_y/stride[0])
    print('Convolution Dimensions: x={} / y={}'.format(x_max, y_max))
    x_count, y_count = 1, 1

    for y in range (0, len_y, stride[0]):
        x_count = 1
        for x in range(0, len_x, stride[0]):
            x_end = x + stride[0]
            y_end = y + stride[0]
            kernel = image[y:y_end, x:x_end]
            #orientation = s1(kernel)
            a = fig.add_subplot(y_max, x_max, (y_count-1)*x_max+x_count)
            a.axis('off')
            plt.imshow(kernel, cmap="gray")
            x_count += 1
        y_count += 1
    plt.show()

"""
END : MNIST IMAGE_UTILS
-----------------------------------------------------------------------------------------------------------------------
"""