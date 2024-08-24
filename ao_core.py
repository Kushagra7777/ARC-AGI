# -*- coding: utf-8 -*-
"""
// aolabs.ai software >ao_core/ao_core.py (C) 2022 Animo Omnis Corporation. All Rights Reserved.

Please note-- this software is not licensed under any open source, freeware, copyleft, public domain, Creative Commons, or any other license other than the terms of the Closed Beta Agreement.
This software is distributed only pursuant to the terms of individually-executed Closed Beta Agreements between Animo Omnis Corporation and a small group of trusted individual Beta testers. The terms of those Agreements prohibit the further distribution of our software.
For more information, please refer to AOLabs.ai/strategy. To help us by alerting us to the distribution of our software without our permission, we would appreciate it if you would contact ali@aolabs.ai.

Thank you for joining our Beta!
"""

# AO Labs Modules
#from .agent.neuron import neuron as na

# 3rd Party Modules
import numpy as np

# ----
# ao_core

## neuron-level code

def fire_prep(training_set_inputs, training_set_outputs):
    #
    # Calculate muk, which is the same for each neuron for a given training set
    #

    num_training_examples = training_set_inputs.shape[0]
    input_size = training_set_inputs.shape[1]

    # training_set_outputs should be a binary array of length num_training_examples
    num_firing_outputs = sum(training_set_outputs)
    num_non_firing_outputs = num_training_examples - num_firing_outputs

    # Duplicate the training-set-outputs to be the same shape as . . . something transposed?
    duplicate_by_rows = 1
    training_set_outputs_duplicated = np.tile(training_set_outputs, (input_size, duplicate_by_rows)).T

    # Create firing and non-firing arrays for calculations
    firing_examples = np.copy(training_set_outputs_duplicated)
    non_firing_examples = np.copy(training_set_outputs_duplicated)

    # In the firing-array set the indexes of non-firing training-set examples to be 99 - i.e. different
    # This leaves the indices of firing training-set examples set to 1
    firing_examples[training_set_outputs_duplicated == 0] = 99
    # For each input row in the training-set:
    #   Count how many of its firing pixels are the same as the training-set-outputs reshaped
    firing_counts = sum(firing_examples == training_set_inputs)
    firing_counts = firing_counts.astype(float)

    # In the non-firing array set the indexes of firing training-set examples to be 99 - i.e. different
    # And set the indices of non-firing training-set examples to be 1 - i.e. the same
    non_firing_examples[training_set_outputs_duplicated == 1] = 99
    non_firing_examples[training_set_outputs_duplicated == 0] = 1
    # For each input row in the training-set:
    #   Count how many of its non-firing pixels are the same as the training-set-outputs reshaped
    non_firing_counts = sum(non_firing_examples == training_set_inputs)
    non_firing_counts = non_firing_counts.astype(float)

    # Set any infinity or other undefined numbers to zero in the counts
    firing_counts[np.isnan(firing_counts)] = 0
    non_firing_counts[np.isnan(non_firing_counts)] = 0

    # Normalize the firing and non-firing counts by the total number of firing and non-firing outputs
    normalized_firing_counts = firing_counts / num_firing_outputs
    normalized_non_firing_counts = non_firing_counts / num_non_firing_outputs

    # Set any infinity or other undefined numbers to zero in the counts
    normalized_firing_counts[np.isnan(normalized_firing_counts)] = 0
    normalized_non_firing_counts[np.isnan(normalized_non_firing_counts)] = 0

    # muk is an array of length num_training_examples
    muk = abs(normalized_firing_counts - normalized_non_firing_counts)

    return muk


def fire(training_set_inputs, training_set_outputs, test_input, muk=[], DD=True, Hamming=True, Default=True):
    # Duplicate the test_input to make it the same shape as training_set_inputs to ease calculations
    duplicate_by_rows = 1
    num_training_examples = training_set_inputs.shape[0]
    test_input_duplicated = np.tile(test_input, (num_training_examples, duplicate_by_rows))
    # Create a zero "No CGA" response
    response = [0, "No CGA"]

    if DD is True:
        response = calc_DD(test_input_duplicated, training_set_inputs, training_set_outputs, muk, response)
    if Hamming is True and is_no_cga_response(response):
        response = calc_hamming(test_input_duplicated, training_set_inputs, training_set_outputs, response)
    if Default is True and is_no_cga_response(response):
        response = [np.random.randint(0,2), "CGA Default"]

    return response


def calc_DD(test_input_duplicated, training_set_inputs, training_set_outputs, muk, default_response):
    count_by_rows = 1
    # For each training-set-input row
    #   Calculate nand with test-input
    #   Vector multiply by muk for that row
    # Then sum the results by row
    dd_distances = np.sum((abs(test_input_duplicated - training_set_inputs) * muk), count_by_rows)

    return get_unique_output_or_default(dd_distances, training_set_outputs, "DD", default_response)


def calc_hamming(test_input_duplicated, training_set_inputs, training_set_outputs, default_response):
    count_by_rows = 1
    # Calc Hamming distance of the (duplicated) test-input from each row of the training-set-inputs
    hamming_distances = np.sum(training_set_inputs != test_input_duplicated, count_by_rows)

    return get_unique_output_or_default(hamming_distances, training_set_outputs, "Hamming", default_response)


def get_unique_output_or_default(distances, training_set_outputs, label, default_response):
    # Get the indices of training-set rows having the minimum distance from test-input
    minimum_distance = np.min(distances)
    minimum_distance_indices = distances == minimum_distance

    # Get the unique set of outputs corresponding to those minimum-distance rows
    outputs_for_minimum_distance = np.unique(training_set_outputs[minimum_distance_indices])

    # If there is a single unique output then use it, else return the default response
    if outputs_for_minimum_distance.size == 1:
        return [outputs_for_minimum_distance[0], label]
    return default_response


def is_no_cga_response(response):
    return response[1] == "No CGA"


def na_flip_prune(training_set_inputs, training_set_outputs, prune_decoupled=False, flip=True):

    # flips the sets, since np.unique prunes copies starting from the top        
    training_set_inputs = np.flip(training_set_inputs, axis=0)
    training_set_outputs = np.flip(training_set_outputs, axis=0)

    # combining both input and output in one array
    training_set_combined = np.zeros( (training_set_inputs.shape[0], training_set_inputs.shape[1]+1) )
    training_set_combined[:, 0:-1] = training_set_inputs
    training_set_combined[:, -1] = training_set_outputs
    
    if prune_decoupled is True: arr = training_set_inputs
    else: arr = training_set_combined
    
    unique_indexes = np.unique( arr, axis=0, return_index=True, return_counts=True)
    unique_indexes = sorted(unique_indexes[1])

    training_set_inputs_final = training_set_inputs[unique_indexes]
    training_set_outputs_final = training_set_outputs[unique_indexes]
    
    if flip is False:
        training_set_inputs_final = np.flip(training_set_inputs_final, axis=0)
        training_set_outputs_final = np.flip(training_set_outputs_final, axis=0)

    transformed = [training_set_inputs_final, training_set_outputs_final]
    
    return transformed


## agent-level code

class Agent(object):
    """docstring for Agent"""
    
    steps = 100000 # used to perallocate self.story and .metastory, the numpy
                   # arrays that record all neural state machine firing activity
    
    def __init__(self, Arch, notes=""):
        super(Agent, self).__init__()
        self.arch = Arch
        self.notes = notes
        
        self.story = np.zeros([self.steps, self.arch.n_total], dtype=int)     # the global (Agent-level) lookup table / history of neuron firing activity (i.e. 0 or 1)
        self.metastory = np.zeros([self.steps, self.arch.n_total], dtype="O")     # same above except metadata (i.e. why the neurons fired the way they did)

        self.story[0, :] = np.random.randint(0, 2, self.arch.n_total)
        self.story[0, self.arch.C__flat] =  0 
        self.metastory[0, :] = "initial"
        self.state = 1     # counter for the overall state of Agent, used to index Agent.story/metastory; starts from 1 because agentsare initialized with a random starting state        
    
        self.neurons = np.zeros(self.arch.n_total, dtype="O")
        self._neuron_creator()
    def _neuron_creator(self):     # thanks Brian Strand (github@CitizenB) for this lesson :)              
        for n in self.arch.IQZC:
            self.neurons[n] = Neuron(n, self)


####### Core Method -- this constitues our top-level API #######
    # broken down into sections as follows:    
    ####### I - input
    ####### C - control
    ### command (labels, forcing positive learning, forcing negative learning)
    ### instincts
    ####### QZ - state and output
    ### controlled neurons, iconic
    ### pain controlled neurons, iconic
    ### uncontrolled neurons, CGA
             
    def next_state(self, INPUT, LABEL=[], INSTINCTS=False, Cneg=False, Cpos=False,
                   print_result=False, sel="", DD=True, Hamming=True, Default=True, unsequenced=False,
                   flip_prune=True, prune_decoupled=True, flip=True):

        INPUT = np.asarray(INPUT, dtype=int)
        LABEL = np.asarray(LABEL, dtype=int)  # might not be necessary. check where to do int instead of float later
        s = self.state
           
####### I neuron firing
        self.story[s, self.arch.I__flat] = INPUT[self.arch.I__flat]
        self.metastory[s, self.arch.I__flat] = "User Input"

####### C neuron firing
    # First, the C-command neurons, which involve user action (either supplying a label,
    # or forcing learning positively [Cpos] or negatively [Cneg] ).
    # Command neurons are usually connected to all neurons (i.e. a label affects all neurons).
### Command neurons
        label_supplied = False
        if LABEL.size != 0:
#            LABEL = np.asarray(LABEL)    # moved trandofmr to ln 213
            label_supplied = True
            c = self.arch.C__flat[0]
            self.story[s, c] = 1
            self.metastory[s, c] = "External Label"
        elif Cpos is True:
            c = self.arch.C__flat[1]
            self.story[s, c] = 1
            self.metastory[s, c] = "C+ reinforce"
        elif Cneg is True:
            c = self.arch.C__flat[2]
            self.story[s, c] = 1
            self.metastory[s, c] = "C- reinforce"            
### Instinct neurons - else any other C neurons fire according to their instinct rules; in control, command takes precedence over instinct
        elif INSTINCTS == True:
            for c in self.arch.C__flat[3:]:
                instinct_rule = self.arch.datamatrix[4, c]
                instinct_response = instinct_rule(INPUT, self)                
                self.story[s, c] = instinct_response[0]
                self.metastory[s, c] = instinct_response[1]       
        C_active = self.arch.C__flat[ self.story[s, self.arch.C__flat]==1 ]     # nids of active C neurons
        C_active_pain = self.arch.C__flat_pain[ self.story[s, self.arch.C__flat_pain]==1 ]
       
####### QZ neuron firing        
##### Iconic Training     # terminology from IA of imperial college, research paper #1
        if C_active.size != 0:     # if there are active C neurons, there is iconic training
            # we extract the nids connected to active C neurons as "_controlled;" we also extract each controlled neuron's dominant connection (dominant = the neuron its fixed to copy when doing iconic transfer)
            # all other neurons are "_noncontrolled"
            Q_controlled, Z_controlled, Dominant_of_controlled_Q, Dominant_of_controlled_Z, Q_noncontrolled, Z_noncontrolled = self._controlled_states(C_active)
###      Firing Q Iconically - Q copies I (they are both same shape, preserving the "Icon" of the event, the precept)
            self.story[s, Q_controlled] = self.story[s, Dominant_of_controlled_Q]
            self.metastory[s, Q_controlled] = "Iconic transfer"
###      Firing Z Iconically - Z copies the past Z (at state-1), since osentsibly it is that past Z action which led to the input which triggered the learning event (eg. the clam had its mouth open, that's why it ate and the c neuron fired, so it keeps its mouth open)
            self.story[s, Z_controlled] = self.story[s-1, Dominant_of_controlled_Z]
            self.metastory[s, Z_controlled] = "Copied Past Action"
##### Iconic Training, Pain - applying the same procedure as above but only for pain C neurons with associated changes
            if C_active_pain.size != 0:    # if some of the active C neurons are pain neurons, then---
                Q_controlled_, Z_controlled_, Dominant_of_controlled_Q_, Dominant_of_controlled_Z_, Q_noncontrolled_, Z_noncontrolled_ = self._controlled_states(C_active_pain)        
###      Firing Q Iconically Pain - Q copies the inverse of I, to avoid  (they are both same shape, preserving the "Icon" of the event, the precept)    
                self.story[s, Q_controlled_] = 1 - self.story[s, Dominant_of_controlled_Q_]
                self.metastory[s, Q_controlled_] = "Iconic transfer - PAIN"      
###      Firing Z Iconically Pain - Z copies the past Z (at state-1), since osentsibly it is that past Z action which led to the input which triggered pleasure (eg. the clam had its mouth open, that's why it ate, felt good c neuron fires)
                self.story[s, Z_controlled_] = 1 - self.story[s-1, Dominant_of_controlled_Z_]
                self.metastory[s, Z_controlled_] = "Copied INVERSE Past Action - Inverse due to PAIN"                
            # LABEL override-- the presence of a label overrides the Z state with a copy of the LABEL; you've *told* the agent what to do; commanded it.
            if label_supplied is True:
                self.story[s, self.arch.Z__flat] = LABEL
                self.metastory[s, self.arch.Z__flat] = "LABEL"

###      Firing all _noncontrolled qz neurons using CGA
            for qz in self.neurons[np.concatenate((Q_noncontrolled, Z_noncontrolled))]:
                noutput = qz.next_state(DD, Hamming, Default)                
                self.story[s, qz.nid] = noutput[0]
                self.metastory[s, qz.nid] = noutput[1]

            # Updating local (neuron) lookup tables (for controlled neurons only)
            for qz in self.neurons[np.concatenate((Q_controlled, Z_controlled))]:
                qz.gen_tsets(flip_prune, prune_decoupled, flip, unsequenced)
                qz.gen_dc_array()
            
            self.state += 1     # end of the state; all neurons have fired
        
##### Local Firing Rules if no Iconic Learning (per neuron)
        else: 
            for n in self.neurons[self.arch.QZ__flat]:                
                noutput = n.next_state(DD, Hamming, Default)
                self.story[s, n.nid] = noutput[0]
                self.metastory[s, n.nid] = noutput[1]
            self.state += 1     # end of the state; all neurons have fired            

        if print_result is True:
            self.print_result(sel)
####### End of Core Method Agent.next_state() #######

    # helper function for next_state
    def _controlled_states(self, C_active):
        # extracting the ids of neurons connected to C neurons (to active C neurons only)
        if C_active.size > 1:               # wtf is going on here. certainly investigate and remove numpy dependency where possible for python list
            QZ_controlled = np.asarray(list(set(np.concatenate(self.arch.datamatrix[3, C_active]))))
        else: QZ_controlled = self.arch.datamatrix[3, C_active][0]
        
        Q_controlled = np.asarray(list( set(QZ_controlled) - set(self.arch.Z__flat)))
        Z_controlled = np.asarray(list( set(QZ_controlled) - set(self.arch.Q__flat)))
        # nids of QZ neurons affected by at least 1 active c

        Dominant_of_controlled_Q = self.arch.datamatrix[4, Q_controlled].astype(int)
        Dominant_of_controlled_Z = self.arch.datamatrix[4, Z_controlled].astype(int)
        # nids of the dominant neurons of the c affected QZ neurons
                
        Q_noncontrolled = np.asarray(list( set(self.arch.Q__flat) - set(Q_controlled) )).astype(int)
        Z_noncontrolled = np.asarray(list( set(self.arch.Z__flat) - set(Z_controlled) )).astype(int)
        # nids of QZ neurons *not* affected by any active C neurons

        return [Q_controlled, Z_controlled, Dominant_of_controlled_Q, Dominant_of_controlled_Z, Q_noncontrolled, Z_noncontrolled]


    def reset_state(self, print_result=False):        
        self.story[self.state, self.arch.IQZ] = np.random.randint(0,2, self.arch.IQZ.size)        
        self.metastory[self.state, self.arch.IQZ] = "Reset (Random)"
        self.state += 1
        if print_result is True:
            self.print_result()


    def zeros_state(self, print_result=False):        
        self.story[self.state, self.arch.IQZ] = np.zeros(self.arch.IQZ.size)        
        self.metastory[self.state, self.arch.IQZ] = "Zero State"
        self.state += 1
        if print_result is True:
            self.print_result()
            

    def ones_state(self, print_result=False):        
        self.story[self.state, self.arch.IQZ] = np.ones(self.arch.IQZ.size)        
        self.metastory[self.state, self.arch.IQZ] = "Ones State"
        self.state += 1        
        if print_result is True:
            self.print_result()
            

    def print_result(self, sel=""):
        if sel == "":
            print("At State # "+str(self.state-1)+"  the Z response was:")
            print(self.story[self.state-1, self.arch.Z__flat])
            print(self.metastory[self.state-1, self.arch.Z__flat])
        if sel != "":
            print("At State # "+str(self.state-1)+"  the selected response was (for neurons"+str(sel)+"):")
            print(self.story[self.state-1, self.arch.IQZC[sel]])
            print(self.metastory[self.state-1, self.arch.IQZC[sel]])


    def _delete_state(self, start_state, end_state):        
        self.story[ start_state : end_state, :] = 0
        self.metastory[ start_state : end_state, :] = 'developer deleted'
        self._update_neuron_data()
        self.state= start_state        
        print("States "+str(start_state)+" to "+str(end_state)+" have been deleted, and neuron data reset.")
        print("State has also been returned to -- state# "+str(start_state))


    def _update_neuron_data(self, flip_prune=True, prune_decoupled=True, flip=True, unsequenced=False):
        """
        necessary only when a user makes manual changes to agent.story.
        agent.story is the global lookup table; each neuron generates their
        own neuron lookup table (in the form of neuron.tsets & neuron.outputs)
        from the global lookup table, and this only happens when a c neuron
        is triggered. So if you change a c active state in agent.story, 
        be sure to run this function or else your changes won't be reflected
        in the neurons
        """
        for n in self.neurons[ self.arch.QZ__flat ]:
            
            if "CGA" in self.arch.datamatrix[0, n.nid]:
                
                n.gen_tsets(flip_prune, prune_decoupled, flip, unsequenced)
                n.gen_dc_array()

                
    def _count_Controls( self, nid_c_neuron=[] ):
       """
       Counts the number of times the control neurons fired (checks all rows that have at least 1 c=1 by default, or nid_c_neuron can specifiy particular c neurons to return the nuber of times they've fired')
       """
       count = self.story[ 0:self.state, self.arch.C__flat ]                        

       if nid_c_neuron != []:
            count = self.story[ 0:self.state, nid_c_neuron ]    
       try:                       
           count = np.sum(count, 1)
           count[ count > 1 ] = 1
       except AttributeError:
           pass
            
       count = sum(count)
       return count
            
            
            
class Neuron:
    """docstring for Neuron"""
    def __init__(self, nid, Agent):
        super(Neuron, self).__init__()
        self.nid = nid
        self.Agent = Agent
        
        
    def gen_tsets(self, flip_prune=True, prune_decoupled=True, flip=True, unsequenced=False):
        # this method generates the training set and output set of a neuron, based on that neuron's connections, from the global lookup table (Agent.story)
        # stores in the neuron 2 objects: tsets, a 2D numpy array in shape (controlled states*2, number of connections) and outputs, a 1D numpy array (controlled states*2)

        n = self.nid
        Agent = self.Agent
        n_type =            Agent.arch.datamatrix[0, n] 
        in_conn =           Agent.arch.datamatrix[1, n] 
        neighbor_conn =     Agent.arch.datamatrix[2, n]
        C_conn =            Agent.arch.datamatrix[3, n]
        numberof_conns = len(in_conn) + len(neighbor_conn)
                       
        ### Preallocating tsets by counting the number of states (rows) of Agent.story with active Cs
        rows_active_Call = Agent.story[0:Agent.state+1, C_conn]
        try:                       
            rows_active_Call = np.sum(rows_active_Call, 1)
            rows_active_Call[ rows_active_Call > 1 ] = 1
        except AttributeError: # in case arch only has 1 C neuron (the default "if label c" neuron)
            pass
        rows = sum(rows_active_Call).astype(int) * 2
        row_index = np.arange(1, rows, 2)
        row_index_prior = np.arange(0, rows, 2)                
        tsets = np.zeros([rows, numberof_conns])
        
        # no doubt this is among the most badly written line all of. Must refactor to not rely on both np.unique and nonzero. it does work though...
        # in summary, here we are extracting from Agent.story the states with active Cs   -- if need to apply a specific rules later: # C_pain_states_indexmask= Agent.story[0:Agent.state+1, C_conn_pain] == 1        
        C_states_index = np.unique(Agent.story[0:Agent.state+1, C_conn].nonzero()[0])
        C_states_index_prior = C_states_index - 1        

        # below tset is populated from Agent.story while keeping the same sequence
        if "Q" in n_type:
            tsets[row_index_prior, 0:len(in_conn)] = Agent.story[C_states_index][:, in_conn]                                       
            tsets[row_index_prior, len(in_conn):] = Agent.story[C_states_index_prior][:, neighbor_conn]
    
            tsets[row_index, 0:len(in_conn)] = Agent.story[C_states_index][:, in_conn]
            tsets[row_index, len(in_conn):] = Agent.story[C_states_index][:, neighbor_conn]   
    
            outputs= np.repeat(Agent.story[C_states_index, n], 2, 0)
            
        if "Z" in n_type:
            tsets[row_index_prior, 0:len(in_conn)] = Agent.story[C_states_index_prior][:, in_conn]
            tsets[row_index_prior, len(in_conn):] = Agent.story[C_states_index_prior-1][:, neighbor_conn]
            
            tsets[row_index, 0:len(in_conn)] = Agent.story[C_states_index][:, in_conn]
            tsets[row_index, len(in_conn):] = Agent.story[C_states_index_prior][:, neighbor_conn]

            outputs= np.repeat(Agent.story[C_states_index, n], 2, 0)
            
            if unsequenced is True: # unsequenced is like the v0.1.0 bClam, where we're not teaching the agent a sequence, we're using Agent.reset_state between next_states. 
                tsets[row_index_prior, 0:len(in_conn)] = Agent.story[C_states_index][:, in_conn]
                tsets[row_index_prior, len(in_conn):] = 1 - Agent.story[C_states_index_prior][:, neighbor_conn]
    
        if flip_prune is True: [tsets, outputs] = na_flip_prune(tsets, outputs, prune_decoupled, flip)

        self.tsets = tsets
        self.outputs = outputs
        
    
    def gen_dc_array(self):
        if self.tsets.size == 0: pass
        else: self.dc_array = fire_prep(self.tsets, self.outputs)
        
        
    def next_state(self, DD, Hamming, Default):

        n = self.nid
        Agent = self.Agent
        # if self.ntype == "CGA":    # need this functionality later for Qaux neurons, ntype "aux," (they could fire according to time / circadian rythm), but not doing that right now. Gotta focus!
        
        try:            
            in_conn =           Agent.arch.datamatrix[1, n] 
            neighbor_conn =     Agent.arch.datamatrix[2, n]
            
            newinput = np.append(Agent.story[Agent.state, in_conn],
                                 Agent.story[Agent.state - 1, neighbor_conn])
            
            result = fire(self.tsets, self.outputs, newinput, self.dc_array, DD, Hamming, Default)
            if result[1] == "No CGA":
                result[0] = Agent.story[Agent.state-1, n]  # if none of the 2 CGA conditons (DD or Hamming) are met or are disable, and the default condition is also disabled, then the neuron copies its passed state (doesn't change state)
            return result
        except AttributeError: # in case neuron does not yet have .tsets, .outputs, or .dc_array attributes (i.e. if it hasn't undergone any iconic training yet)
            return [np.random.randint(0,2), "Untrained (random)"]
