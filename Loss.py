import argparse
import numpy as np
import torch


class Loss():
    def __init__(self, 
                 constraint_coeff_mu = [1, 1e-06], 
                 constraint_coeff_mu_update_factors = [1.02, 1.02], 
                 fair_loss_p = 1,
                 budget = args.budget,
                 N,
                 constraint_epoch_start_schedule = [0, 100, 200],
                 constraint_epsilon = [0.0, 0.0],
                 temp_decay_factor = 0.999,
                 model):
        

        self.constraint_epoch_start_schedule = constraint_epoch_start_schedule
        self.constraint_epsilon = constraint_epsilon
        self.constraint_coeff_mu = constraint_coeff_mu

        self.flow_loss_switch = 1
        self.fair_loss_switch = 1
        self.budget_loss_switch = 1
        
        self.fair_loss_mu = 0
        self.edit_loss_mu = 0
        
        self.fair_loss_mu_update_factor = constraint_coeff_mu_update_factors[0]
        self.edit_loss_mu_update_factor = constraint_coeff_mu_update_factors[1]
        
        self.fair_loss_lambda = 0
        self.edit_loss_lambda = 0

        self.fair_loss_p = fair_loss_p
        
        self.budget = budget
        self.N = N

        self.temp_decay_factor = temp_decay_factor

        self.model = model

    def on_epoch_end(self, epoch):
        if epoch>self.constraint_epoch_start_schedule[0]:
            self.fair_loss_mu = self.constraint_coeff_mu[0]
        
        if epoch%2==0:
            self.flow_loss_switch = 1
            self.fair_loss_switch = 0
            self.budget_loss_switch = 0
        else:
            self.flow_loss_switch = 0
            self.fair_loss_switch = 1
            self.budget_loss_switch = 0
            
            
        if epoch>self.constraint_epoch_start_schedule[1]:
            self.edit_loss_mu = self.constraint_coeff_mu[1]
            if epoch%3==0:
                self.budget_loss_switch = 1
                self.flow_loss_switch = 0
                self.fair_loss_switch = 0

            
    def on_batch_end(self, epoch):
        global temp
        if epoch>self.constraint_epoch_start_schedule[0] or epoch>self.constraint_epoch_start_schedule[1]:
            outputs = model(next(loader), temp)
            
        if epoch>self.constraint_epoch_start_schedule[0]:
            # Learning proxy lagrangian for fair loss
            vs = outputs[0]
            Vs = [np.mean(vs[i]) for i in range(len(vs))]
            fairlosses = np.sum(np.abs(Vs - np.mean(Vs)))**self.fair_loss_p
            fairlosses = 2 * self.fair_loss_mu * fairlosses
            self.fair_loss_lambda = (self.fair_loss_lambda + fairlosses)/2
        
        if epoch>self.constraint_epoch_start_schedule[1]:
            # Learning proxy lagrangian for edit loss
            Es = outputs[1]
            edit_loss = 2 * self.edit_loss_mu * np.mean(np.sum(Es, axis=(-1)) - self.budget)
            self.edit_loss_lambda = self.edit_loss_lambda + edit_loss
        
        if epoch>self.constraint_epoch_start_schedule[2]:
            global temp
            # Learning proxy lagrangian for edit loss
            temp *= self.temp_decay_factor

    def fair_flow_loss(self, vs):
        return self.param_schedule.flow_loss_switch * self.flow_loss(vs) + self.param_schedule.fair_loss_switch * self.weighted_fair_loss(vs)

    def flow_loss(self, vs):
        return -1 * (torch.mean(vs, dim=-1))**self.main_obj_optimizer_p

    def weighted_fair_loss(self, vs): 
        #fair_loss = self.fair_loss_spectral_norm(y_true, y_pred)
        fair_loss = self.fair_loss(vs)
        fair_loss = torch.clip(fair_loss,self.constraint_epsilon[0],1e20)-self.constraint_epsilon[0]
        fair_loss = self.param_schedule.fair_loss_mu * (fair_loss**2) + 2 * self.param_schedule.fair_loss_lambda * fair_loss
        
        return fair_loss

    def budget_loss(self, R):
        edit_loss = self.num_edits_exceeded(R)
        edit_loss = torch.clip(edit_loss, self.constraint_epsilon[1] , 1e20) - self.constraint_epsilon[1]
        edit_loss = self.param_schedule.edit_loss_mu * (edit_loss**2) + 2 * self.param_schedule.edit_loss_lambda * edit_loss
        
        return edit_loss * self.param_schedule.budget_loss_switch

    def num_edits_exceeded(self, R):
        print('self.budget:',self.budget)
        edits_exceeded = torch.clip(torch.sum(y_pred, dim=(-1)) - self.budget, 0 , 1e20)
        return edits_exceeded

    def loss_function(self, vs, R):
        
        fair_flow = fair_flow_loss(vs)
        budget = budget_loss(R)
        return fair_flow+budget


