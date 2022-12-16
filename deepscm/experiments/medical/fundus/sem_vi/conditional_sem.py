import torch
import pyro
from pyro.nn import pyro_method
from pyro.distributions import Normal, Bernoulli, TransformedDistribution, Gumbel, Categorical
from pyro.nn import DenseNN
from deepscm.experiments.medical.fundus.sem_vi.base_sem_experiment import BaseVISEM, MODEL_REGISTRY
from torch.distributions.utils import logits_to_probs


class ConditionalVISEM(BaseVISEM):
    # context_dim = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        # T2D flow
        self.T2D_net = DenseNN(2, [8, 16], param_dims=[2], nonlinearity=torch.nn.LeakyReLU(.1))
        pyro.module("T2D_net", self.T2D_net, update_module_params=True)
    @pyro_method
    def pgm_model(self):
        sex_logits = pyro.param("sex_logits", self.sex_logits)
        sex_dist = Bernoulli(logits=sex_logits).to_event(1)
        sex = pyro.sample('sex', sex_dist)
        
        age_base_dist = Normal(self.age_base_loc, self.age_base_scale).to_event(1)
        age_dist = TransformedDistribution(age_base_dist, self.age_flow_transforms)
        age = pyro.sample('age', age_dist)
        age_ = self.age_flow_constraint_transforms.inv(age)
        # pseudo call to thickness_flow_transforms to register with pyro
        _ = self.age_flow_components

        T2D_context = torch.cat([sex.cuda(), age_.cuda()], dim=-1)
        T2D_logits = self.T2D_net(T2D_context.unsqueeze(0)).squeeze(0)
        T2D_base_dist = Gumbel(loc=self.gumbel_loc, scale=self.gumbel_scale).to_event(1)
        T2D_base = pyro.sample("T2D_base", T2D_base_dist)
        T2D_prob = logits_to_probs(T2D_logits + T2D_base)
        T2D_dist = Categorical(T2D_prob)
        T2D = pyro.sample('T2D', T2D_dist)

        return age, sex, T2D


    @pyro_method
    def model(self):
        age, sex, T2D = self.pgm_model()
        age_ = self.age_flow_constraint_transforms.inv(age)
        sex = sex
        T2D = T2D.unsqueeze(-1)

        # this is a patch to fix the unmatched shapes of T2D and other variables
        if len(T2D.shape) == len(age.shape)+1:
            T2D = T2D.squeeze(-1)

        z = pyro.sample('z', Normal(self.z_loc, self.z_scale).to_event(1))

        latent = torch.cat([z.cuda(), age_.cuda(), sex.cuda()], 1)

        x_dist = self._get_transformed_x_dist(latent)
        x = pyro.sample('x', x_dist)

        return x, z, age, sex, T2D

    @pyro_method
    def guide(self, x, age, sex, T2D):
        with pyro.plate('observations', x.shape[0]):
            hidden = self.encoder(x)

            age_ = self.age_flow_constraint_transforms.inv(age)
            sex = sex
            T2D = T2D.unsqueeze(-1)

            hidden = torch.cat([hidden.cuda(), age_.cuda(), sex.cuda()], 1)

            latent_dist = self.latent_encoder.predict(hidden)
            z = pyro.sample('z', latent_dist)

        return z


MODEL_REGISTRY[ConditionalVISEM.__name__] = ConditionalVISEM
