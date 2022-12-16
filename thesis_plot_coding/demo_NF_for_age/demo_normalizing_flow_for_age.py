import torch
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import pandas as pd
from torch.utils.data import DataLoader
from pyro.poutine import trace, condition
from torch.distributions import Independent
from matplotlib import pyplot as plt
import seaborn as sns
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.transforms import (
    ComposeTransform, AffineTransform, ExpTransform, Spline
)

if torch.cuda.is_available():
  dev = "cuda:2"
else:
  dev = "cpu"

print(dev)
device = torch.device(dev)


train_dir = '/mnt/beta/sliao/DSCM_fundus/deepscm/deepscm/datasets/split_dir/train_features.xlsx'
val_dir = '/mnt/beta/sliao/DSCM_fundus/deepscm/deepscm/datasets/split_dir/val_features.xlsx'

dtype_dic = {'Filename': 'str', 'RandomID':'str', 'Age': 'int', 'Sex': 'int', 'T2D': 'int', 'HT': 'int'}
df = pd.read_excel(train_dir, engine='openpyxl', dtype=dtype_dic)
metrics = {col: df[col] if col == 'Filename' or col == 'RandomID' else torch.as_tensor(df[col]).float() for col in df.columns}

df_val = pd.read_excel(val_dir, engine='openpyxl', dtype=dtype_dic)
metrics_val = {col: df_val[col] if col == 'Filename' or col == 'RandomID' else torch.as_tensor(df_val[col]).float() for col in df_val.columns}



#train
def train(model, guide, dataloader, lr=0.0005, epochs=500):
    pyro.clear_param_store()
    adam_params = {"lr": lr}
    adam = pyro.optim.Adam(adam_params)
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    n_steps = dataloader.dataset.size(0)//dataloader.batch_size

    model_trace_storage={}
    for epoch in range(epochs):
        losses=0.
        for step in range(n_steps):
            loss = svi.step( next(iter(dataloader)))/torch.tensor(64.)
            losses = losses + loss
            m_step = epoch*n_steps +step
            model_trace_storage['step{}'.format(m_step)] = model_trace(model, next(iter(dataloader)))
        if epoch % 10 == 0:
            loss_mean=losses/n_steps
            print('[epoch {}],  iter 317,  loss: {:.4f}'.format(epoch, loss_mean))

    return model_trace_storage

def evaluate(model, dataloader, epochs=100):

    n_steps = dataloader.dataset.size(0)//dataloader.batch_size
    model_trace_storage={}
    for epoch in range(epochs):
        for step in range(n_steps):
            model_trace_storage['step{}'.format(step)] = model_trace(model, next(iter(dataloader)))

    return model_trace_storage


def model_trace(model, data):
    with torch.no_grad():
        m_trace = pyro.poutine.trace(model).get_trace(data)
        m_trace.compute_log_prob()
        model_trace_storage_step={}
        for name, site in m_trace.nodes.items():
            if site["type"] == "sample" and site["is_observed"] ==True:
                fn = site['fn']
                if isinstance(fn, Independent):
                    fn = fn.base_dist

                model_trace_storage_step[name] = {}
                model_trace_storage_step[name]['fn'] = fn
                model_trace_storage_step[name]['log_prob'] = site["log_prob"].mean()
                model_trace_storage_step[name]['loss'] = site["log_prob_sum"]
        return model_trace_storage_step

def vis_log_prob(log_prob, name):
    # plt.xlim(0, 100)
    plt.xlabel('step')
    plt.ylabel('log_prob({})'.format(name))
    plt.plot(log_prob)
    plt.savefig('/mnt/beta/sliao/DSCM_fundus/deepscm/deepscm/datasets/composable_model/{}.png'.format(name))

def vis_transform(tr_age, name):
    fig, axs = plt.subplots(nrows= 6, figsize=(3, 10), constrained_layout=True)
    age_base_sample =tr_age.nodes['age_base']['value'].squeeze(-1).squeeze(-1).detach().numpy()
    sns.kdeplot(age_base_sample, ax=axs[0])
    axs[0].set_title("age_base")
    axs[0].set_xlabel("age_value")

    age_spline_sample =tr_age.nodes['age_spline']['value'].squeeze(-1).squeeze(-1).detach().numpy()
    sns.kdeplot(age_spline_sample, ax=axs[1])
    axs[1].set_title("age_spline")
    axs[1].set_xlabel("age_value")

    age_affine_sample =tr_age.nodes['age_affine']['value'].squeeze(-1).squeeze(-1).detach().numpy()
    sns.kdeplot(age_affine_sample, ax=axs[2])
    axs[2].set_title("age_affine")
    axs[2].set_xlabel("age_value")

    age_exp_sample =tr_age.nodes['age_exp']['value'].squeeze(-1).squeeze(-1).detach().numpy()
    sns.kdeplot(age_exp_sample, ax=axs[3])
    axs[3].set_title("age_exp")
    axs[3].set_xlabel("age_value")

    age_sample =tr_age.nodes['age']['value'].squeeze(-1).squeeze(-1).detach().numpy()
    sns.kdeplot(age_sample, ax=axs[4])
    axs[4].set_title("age_obs")
    axs[4].set_xlabel("age_value")

    sns.kdeplot(metrics['age'], ax=axs[5])
    axs[5].set_title("age_data")
    axs[5].set_xlabel("age_value")

    plt.savefig('/mnt/beta/sliao/DSCM_fundus/deepscm/deepscm/datasets/composable_model/{}_transform.png'.format(name))


## normal spline
age_flow_lognorm = AffineTransform(loc=metrics['age'].log().mean().float().cuda(), scale=metrics['age'].log().std().float().cuda())
def model(data):
    age_flow_components = ComposeTransformModule([Spline(torch.tensor(1))]).cuda()
    age_flow_constraint_transforms = ComposeTransform([age_flow_lognorm, ExpTransform()])
    age_flow_transforms = ComposeTransform([age_flow_components, age_flow_constraint_transforms])

    pyro.module("age_flow_components", age_flow_components)
    with pyro.plate("data", data.size(0)):
        # age base distribution
        age_base_dist = dist.Normal(loc=torch.tensor([0.]).cuda(), scale=torch.tensor([1.]).cuda()).to_event(1)
        age_base =pyro.sample('age_base', age_base_dist)
        # step1 spline
        age_dist_spline = dist.TransformedDistribution(age_base_dist, age_flow_components)
        age_spline =pyro.sample('age_spline', age_dist_spline)
         # step2 affine
        age_dist_affine = dist.TransformedDistribution(age_dist_spline, age_flow_lognorm)
        age_affine =pyro.sample('age_affine', age_dist_affine)
         # step3 exp
        age_dist_exp = dist.TransformedDistribution(age_dist_affine, ExpTransform())
        age_exp =pyro.sample('age_exp', age_dist_exp)

        # age NF flow
        age_dist = dist.TransformedDistribution(age_base_dist, age_flow_transforms)

        pyro.sample("age", age_dist, obs=data)
def guide(data):
    pass

dataloader_age = DataLoader(metrics['age'].unsqueeze(1).cuda(), batch_size=64, shuffle=True)

age_trace = train(model, guide, dataloader_age)
log_prob = [age_trace['step{}'.format(i)]['age']['log_prob'] for i in range(len(age_trace))]
vis_log_prob(log_prob, 'age_normal_spline_train')



dataloader_age_val = DataLoader(metrics_val['age'].unsqueeze(1).cuda(), batch_size=64, shuffle=True)
age_eval_trace = evaluate(model, dataloader_age_val)
log_prob = [age_trace['step{}'.format(i)]['age']['log_prob'] for i in range(len(age_eval_trace))]
vis_log_prob(log_prob, 'age_normal_spline_eval')

torch.save(model, '/mnt/beta/sliao/DSCM_fundus/deepscm/deepscm/datasets/composable_model/age_normal_spline.pt')

tr_age = trace(model).get_trace(metrics_val['age'].unsqueeze(1))
vis_transform(tr_age, 'age_normal_spline')

x = torch.linspace(-5.0, 5.0, 1000).unsqueeze(-1)
splined =tr_age.nodes['age_spline']['fn'].transforms[0](x)
plt.plot(x.squeeze(-1),  splined.squeeze(-1).detach().numpy())
plt.savefig('/mnt/beta/sliao/DSCM_fundus/deepscm/deepscm/datasets/composable_model/spline.png')