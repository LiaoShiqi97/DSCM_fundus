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

def vis_transform(tr_age, name):
    fig, axs = plt.subplots(nrows= 6, figsize=(3, 10), constrained_layout=True)
    age_base_sample =tr_age.nodes['age_base']['value'].squeeze(-1).squeeze(-1).cpu().detach().numpy()
    sns.kdeplot(age_base_sample, ax=axs[0])
    axs[0].set_title("age_base")
    axs[0].set_xlabel("age_value")

    age_spline_sample =tr_age.nodes['age_spline']['value'].squeeze(-1).squeeze(-1).cpu().detach().numpy()
    sns.kdeplot(age_spline_sample, ax=axs[1])
    axs[1].set_title("age_spline")
    axs[1].set_xlabel("age_value")

    age_affine_sample =tr_age.nodes['age_affine']['value'].squeeze(-1).squeeze(-1).cpu().detach().numpy()
    sns.kdeplot(age_affine_sample, ax=axs[2])
    axs[2].set_title("age_affine")
    axs[2].set_xlabel("age_value")

    age_exp_sample =tr_age.nodes['age_exp']['value'].squeeze(-1).squeeze(-1).cpu().detach().numpy()
    sns.kdeplot(age_exp_sample, ax=axs[3])
    axs[3].set_title("age_exp")
    axs[3].set_xlabel("age_value")

    age_sample =tr_age.nodes['age']['value'].squeeze(-1).squeeze(-1).cpu().detach().numpy()
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
model = torch.load('/mnt/beta/sliao/DSCM_fundus/deepscm/deepscm/datasets/composable_model/age_normal_spline.pt')

tr_age = trace(model).get_trace(metrics_val['age'].unsqueeze(1).cuda())
# vis_transform(tr_age, 'age_normal_spline')
# plt.show()

fig = plt.figure(figsize=(9, 6))
x = torch.linspace(-5.0, 5.0, 1000).unsqueeze(-1)
splined =tr_age.nodes['age_spline']['fn'].transforms[0](x.cuda())
plt.plot(x.squeeze(-1),  splined.squeeze(-1).cpu().detach().numpy())
plt.title("spline")
plt.ylabel("y")
plt.xlabel("x")
plt.savefig('/mnt/beta/sliao/DSCM_fundus/deepscm/deepscm/datasets/composable_model/spline.png')
plt.close()