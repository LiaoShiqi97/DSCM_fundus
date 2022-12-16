import pyro

from pyro.nn import PyroModule, pyro_method

from pyro.distributions import TransformedDistribution
from pyro.infer.reparam.transform import TransformReparam
from torch.distributions import Independent

from deepscm.datasets.fundus import fundusDataset
from pyro.distributions.transforms import ComposeTransform, SigmoidTransform, AffineTransform

import torchvision.utils
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import os
from functools import partial

EXPERIMENT_REGISTRY = {}
MODEL_REGISTRY = {}


class BaseSEM(PyroModule):
    def __init__(self, preprocessing: str = 'realnvp', downsample: int = -1):
        super().__init__()

        self.downsample = downsample
        self.preprocessing = preprocessing

    def _get_preprocess_transforms(self):
        alpha = 0.05
        num_bits = 8

        if self.preprocessing == 'glow':
            # Map to [-0.5,0.5]
            a1 = AffineTransform(-0.5, (1. / 2 ** num_bits))
            preprocess_transform = ComposeTransform([a1])
        elif self.preprocessing == 'realnvp':
            # Map to [0,1]
            a1 = AffineTransform(0., (1. / 2 ** num_bits))

            # Map into unconstrained space as done in RealNVP
            a2 = AffineTransform(alpha, (1 - alpha))

            s = SigmoidTransform()

            preprocess_transform = ComposeTransform([a1, a2, s.inv])

        return preprocess_transform

    @pyro_method
    def pgm_model(self):
        raise NotImplementedError()

    @pyro_method
    def modelmodel(self):
        raise NotImplementedError()

    @pyro_method
    def pgm_scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg['fn'], TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.pgm_model, config=config)(*args, **kwargs)

    @pyro_method
    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg['fn'], TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    @pyro_method
    def sample(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.model()

        return (*samples,)

    @pyro_method
    def sample_scm(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.scm()

        return (*samples,)

    @pyro_method
    def infer_e_x(self, *args, **kwargs):
        raise NotImplementedError()

    @pyro_method
    def infer_exogeneous(self, **obs):
        # assuming that we use transformed distributions for everything:
        cond_sample = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_sample).get_trace(obs['x'].shape[0])

        output = {}
        for name, node in cond_trace.nodes.items():
            if 'fn' not in node.keys():
                continue

            fn = node['fn']
            if isinstance(fn, Independent):
                fn = fn.base_dist
            if isinstance(fn, TransformedDistribution):
                output[name + '_base'] = ComposeTransform(fn.transforms).inv(node['value'])

        return output

    @pyro_method
    def infer(self, **obs):
        raise NotImplementedError()

    @pyro_method
    def counterfactual(self, obs, condition=None):
        raise NotImplementedError()

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--preprocessing', default='realnvp', type=str, help="type of preprocessing (default: %(default)s)", choices=['realnvp', 'glow'])
        parser.add_argument('--downsample', default=4, type=int, help="downsampling factor (default: %(default)s)")

        return parser


class BaseCovariateExperiment(pl.LightningModule):
    def __init__(self, hparams, pyro_model: BaseSEM):
        super().__init__()

        self.pyro_model = pyro_model

        hparams.experiment = self.__class__.__name__
        hparams.model = pyro_model.__class__.__name__
        self.hparams = hparams
        self.train_batch_size = hparams.train_batch_size
        self.test_batch_size = hparams.test_batch_size
        self.image_index = torch.Tensor([0,1,2,3,4,5,6,7]).long()

        if hasattr(hparams, 'num_sample_particles'):
            self.pyro_model._gen_counterfactual = partial(self.pyro_model.counterfactual, num_particles=self.hparams.num_sample_particles)
        else:
            self.pyro_model._gen_counterfactual = self.pyro_model.counterfactual

        if hparams.validate:
            import random

            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.autograd.set_detect_anomaly(self.hparams.validate)
            pyro.enable_validation()

    def prepare_data(self):
        downsample = None if self.hparams.downsample == -1 else self.hparams.downsample
        train_crop_type = self.hparams.train_crop_type if hasattr(self.hparams, 'train_crop_type') else 'random'
        split_dir = self.hparams.split_dir if hasattr(self.hparams, 'split_dir') else '/mnt/beta/sliao/DSCM_fundus/deepscm/deepscm/datasets/dataset_selected_feature/'
        data_dir = self.hparams.data_train_dir if hasattr(self.hparams, 'data_dir') else '/mnt/alpha/diabetes/MS/data/images_original/'

        self.fundus_train = fundusDataset(f'{split_dir}train_features.xlsx', base_path=data_dir, crop_type=train_crop_type, downsample=downsample)  # noqa: E501
        self.fundus_val = fundusDataset(f'{split_dir}val_features.xlsx', base_path=data_dir, crop_type='center', downsample=downsample)
        self.fundus_test = fundusDataset(f'{split_dir}test_features.xlsx', base_path=data_dir, crop_type='center', downsample=downsample)

        self.torch_device = self.trainer.root_gpu if self.trainer.on_gpu else self.trainer.root_device

        # TODO: change ranges and decide what to condition on
        sex= torch.tensor([0., 1.])
        self.sex_range = sex.repeat(2).unsqueeze(1)

        T2D= torch.tensor([0., 1.])
        self.T2D_range= T2D.repeat_interleave(2).unsqueeze(1)

        self.z_range = torch.randn([1, self.hparams.latent_dim], device=self.torch_device, dtype=torch.float).repeat((4, 1))

        self.pyro_model.age_flow_lognorm_loc = (self.fundus_train.metrics['age'].log().mean().to(self.torch_device).float())
        self.pyro_model.age_flow_lognorm_scale = (self.fundus_train.metrics['age'].log().std().to(self.torch_device).float())

        if self.hparams.validate:
           print(f'set age_flow_lognorm {self.pyro_model.age_flow_lognorm.loc} +/- {self.pyro_model.age_flow_lognorm.scale}')

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.fundus_train, batch_size=self.train_batch_size, num_workers= 48, shuffle=True)

    def val_dataloader(self):
        self.val_loader = DataLoader(self.fundus_val, batch_size=self.test_batch_size, num_workers=48, shuffle=False)
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(self.fundus_test, batch_size=self.test_batch_size, num_workers=48, shuffle=False)
        return self.test_loader

    def forward(self, *args, **kwargs):
        pass

    def prep_batch(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_epoch_end(self, outputs):
        outputs = self.assemble_epoch_end_outputs(outputs)

        metrics = {('val/' + k): v for k, v in outputs.items()}

        if self.current_epoch % self.hparams.sample_img_interval == 0:
            self.sample_images()

        self.log_dict(metrics)

    def test_epoch_end(self, outputs):
        print('Assembling outputs')
        outputs = self.assemble_epoch_end_outputs(outputs)

        samples = outputs.pop('samples')

        sample_trace = pyro.poutine.trace(self.pyro_model.sample).get_trace(self.hparams.test_batch_size)
        samples['unconditional_samples'] = {
            'x': sample_trace.nodes['x']['value'].cpu(),
            'age': sample_trace.nodes['age']['value'].cpu(),
            'sex': sample_trace.nodes['sex']['value'].cpu(),
            'T2D': sample_trace.nodes['T2D']['value'].unsqueeze(-1).cpu()
        }

        cond_data = {
            'sex': self.sex_range.repeat(self.hparams.test_batch_size, 1),
            'T2D': self.T2D_range.repeat(self.hparams.test_batch_size, 1).squeeze(-1),
            'z': torch.randn([self.hparams.test_batch_size, self.hparams.latent_dim], device=self.torch_device, dtype=torch.float).repeat_interleave(4, 0)
        }
        sample_trace = pyro.poutine.trace(pyro.condition(self.pyro_model.sample, data=cond_data)).get_trace(4 * self.hparams.test_batch_size)
        samples['conditional_samples'] = {
            'x': sample_trace.nodes['x']['value'].cpu(),
            'age': sample_trace.nodes['age']['value'].cpu(),
            'sex': sample_trace.nodes['sex']['value'].cpu(),
            'T2D': sample_trace.nodes['T2D']['value'].unsqueeze(-1).cpu()
        }

        print(f'Got samples: {tuple(samples.keys())}')

        metrics = {('test/' + k): v for k, v in outputs.items()}

        for k, v in samples.items():
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{k}.pt')

            print(f'Saving samples for {k} to {p}')

            torch.save(v, p)

        p = os.path.join(self.trainer.logger.experiment.log_dir, 'metrics.pt')
        torch.save(metrics, p)

        self.log_dict(metrics)
        self.log_dict(metrics)

    def assemble_epoch_end_outputs(self, outputs):
        num_items = len(outputs)

        def handle_row(batch, assembled=None):
            if assembled is None:
                assembled = {}

            for k, v in batch.items():
                if k not in assembled.keys():
                    if isinstance(v, dict):
                        assembled[k] = handle_row(v)
                    elif isinstance(v, float):
                        assembled[k] = v
                    elif np.prod(v.shape) == 1:
                        assembled[k] = v.cpu()
                    else:
                        assembled[k] = v.cpu()
                else:
                    if isinstance(v, dict):
                        assembled[k] = handle_row(v, assembled[k])
                    elif isinstance(v, float):
                        assembled[k] += v
                    elif np.prod(v.shape) == 1:
                        assembled[k] += v.cpu()
                    else:
                        assembled[k] = torch.cat([assembled[k], v.cpu()], 0)

            return assembled

        assembled = {}
        for _, batch in enumerate(outputs):
            assembled = handle_row(batch, assembled)

        for k, v in assembled.items():
            if (hasattr(v, 'shape') and np.prod(v.shape) == 1) or isinstance(v, float):
                assembled[k] /= num_items

        return assembled


    def get_counterfactual_conditions(self, batch):
        counterfactuals = {
            'do(T2D=0)': {'T2D': torch.zeros_like(batch['T2D']).squeeze(-1)},
            'do(T2D=1)': {'T2D': torch.ones_like(batch['T2D']).squeeze(-1)},
            'do(age=40)': {'age': torch.ones_like(batch['age']) * 40},
            'do(age=60)': {'age': torch.ones_like(batch['age']) * 60},
            'do(age=80)': {'age': torch.ones_like(batch['age']) * 80},
            'do(sex=0)': {'sex': torch.zeros_like(batch['sex'])},
            'do(sex=1)': {'sex': torch.ones_like(batch['sex'])},
            'do(age=40, sex=1)': {'age': torch.ones_like(batch['age']) * 40,
                                                              'sex': torch.ones_like(batch['sex'])},
            'do(age=80, sex=0)': {'age': torch.ones_like(batch['age']) * 80,
                                                                 'sex': torch.zeros_like(batch['sex'])}
        }

        return counterfactuals

    def build_test_samples(self, batch):
        samples = {}
        samples['reconstruction'] = {'x': self.pyro_model.reconstruct(**batch, num_particles=self.hparams.num_sample_particles)}

        counterfactuals = self.get_counterfactual_conditions(batch)

        for name, condition in counterfactuals.items():
            samples[name] = self.pyro_model._gen_counterfactual(obs=batch, condition=condition)

        return samples

    def log_img_grid(self, tag, imgs, normalize=True, save_img=False, **kwargs):
        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            torchvision.utils.save_image(imgs, p, normalize=normalize)
        grid = torchvision.utils.make_grid(imgs, normalize=normalize, **kwargs)
        self.logger.experiment.add_image(tag, grid, self.current_epoch)

    def get_batch(self, loader):
        batch = next(iter(self.val_loader))
        if self.trainer.on_gpu:

            batch = self.trainer.accelerator_backend.to_device(batch = batch)
        return batch

    def log_kdes(self, tag, data, save_img=False):
        def np_val(x):
            return x.cpu().numpy().squeeze() if isinstance(x, torch.Tensor) else x.squeeze()

        fig, ax = plt.subplots(1, len(data), figsize=(5 * len(data), 5), sharey=True)
        for i, (name, covariates) in enumerate(data.items()):
            try:
                if len(covariates) == 1:
                    (x_n, x), = tuple(covariates.items())
                    if x_n =='sex':
                        df = pd.DataFrame(np_val(x), columns=['sex'])
                        sns.countplot(x='sex', data=df, ax=ax[i])
                    elif x_n =='T2D':
                        df = pd.DataFrame(np_val(x), columns=['T2D'])
                        sns.countplot(x='T2D', data=df, ax=ax[i])
                    elif x_n =='age':
                        sns.kdeplot(x=np_val(x), ax=ax[i], shade=True, thresh=0.05)
                        ax[i].set_xlabel('age')
                elif len(covariates) == 3:
                    covariates = {name: np_val(value) for (name, value) in tuple(covariates.items())}
                    df = pd.DataFrame.from_dict(covariates)
                    sns.violinplot(x="sex", y="age", hue="T2D", data=df, ax=ax[i])
                    sns.swarmplot(x="sex", y="age", hue="T2D", data=df, dodge=True, size=3, ax=ax[i], palette="pastel")

                else:
                    raise ValueError(f'got too many values: {len(covariates)}')
            except np.linalg.LinAlgError:
                print(f'got a linalg error when plotting {tag}/{name}')

            ax[i].set_title(name)

        sns.despine()

        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            plt.savefig(p, dpi=300)

        self.logger.experiment.add_figure(tag, fig, self.current_epoch)

    def build_reconstruction(self, x, age, sex, T2D, tag='reconstruction'):
        obs = {'x': x, 'age': age, 'sex': sex, 'T2D': T2D}

        recon = self.pyro_model.reconstruct(**obs, num_particles=self.hparams.num_sample_particles)
        self.log_img_grid(tag, torch.cat([x, recon], 0), save_img=True)
        self.logger.experiment.add_scalar(f'{tag}/mse', torch.mean(torch.square(x - recon).sum((1, 2, 3))), self.current_epoch)

    def build_counterfactual(self, tag, obs, conditions, absolute2=None):
        _required_data = ('x', 'age', 'sex', 'T2D')
        assert set(obs.keys()) == set(_required_data), 'got: {}'.format(tuple(obs.keys()))

        imgs = obs['x']
        imgs_show = [torch.index_select(imgs, 0, self.image_index.cuda())]

        # TODO: decide which kde's to plot in which configuration
        if absolute2 == 'sex':
            sampled_kdes = {'orig': {'sex': obs['sex']}}
        elif absolute2 == 'age':
            sampled_kdes = {'orig': {'age': obs['age']}}
        elif absolute2 == 'T2D':
            sampled_kdes = {'orig': {'T2D': obs['T2D'].unsqueeze(-1)}}
        else:
            sampled_kdes = {'orig': {'T2D': obs['T2D'].unsqueeze(-1), 'sex': obs['sex']}, 'age': obs['age']}

        for name, data in conditions.items():
            counterfactual = self.pyro_model._gen_counterfactual(obs=obs, condition=data)

            counter = counterfactual['x']
            sampled_sex = counterfactual['sex']
            sampled_T2D = counterfactual['T2D']
            sampled_age = counterfactual['age']

            # imgs.append(counter)
            imgs_show.append(torch.index_select(counter, 0, self.image_index.cuda()))

            if absolute2 == 'T2D':
                sampled_kdes[name] = {'T2D': sampled_T2D.unsqueeze(-1)}
            elif absolute2 == 'sex':
                sampled_kdes[name] = {'sex': sampled_sex}
            elif absolute2 == 'age':
                sampled_kdes[name] = {'age': sampled_age}
            else:
                sampled_kdes[name] = {'T2D': sampled_T2D, 'sex': sampled_sex, 'age': sampled_age}

        self.log_img_grid(tag, torch.cat(imgs_show, 0), save_img=False)
        # self.log_img_grid(tag, torch.cat(imgs, 0), save_img=True)
        self.log_kdes('{}_{}_sampled'.format(tag, absolute2), sampled_kdes, save_img=False)

    def sample_images(self):
        with torch.no_grad():
            # TODO: redo all this....

            #sample
            sample_trace = pyro.poutine.trace(self.pyro_model.sample).get_trace(self.hparams.test_batch_size)
            samples = sample_trace.nodes['x']['value']
            sampled_sex = sample_trace.nodes['sex']['value']
            sampled_T2D = sample_trace.nodes['T2D']['value'].unsqueeze(-1)
            sampled_age = sample_trace.nodes['age']['value']
            self.log_img_grid('samples', samples.data[:8], save_img=False)

            #cond_sample
            cond_data = {'sex': self.sex_range, 'T2D': self.T2D_range.squeeze(-1), 'z': self.z_range}
            cond_samples, *_ = pyro.condition(self.pyro_model.sample, data=cond_data)(4)
            self.log_img_grid('cond_samples', cond_samples.data, nrow=2, save_img=False)

            #obs_batch
            obs_batch = self.prep_batch(self.get_batch(self.val_loader))
            kde_data = {
                'batch': {'sex': obs_batch['sex'], 'T2D': obs_batch['T2D'].unsqueeze(-1), 'age': obs_batch['age']},
                'sampled': {'sex': sampled_sex, 'T2D': sampled_T2D.unsqueeze(-1), 'age': sampled_age}
            }
            self.log_kdes('sample_kde', kde_data, save_img=False)

            # counterfactual exgeneous
            exogeneous = self.pyro_model.infer(**obs_batch)

            for (tag, val) in exogeneous.items():
                self.logger.experiment.add_histogram(tag, val, self.current_epoch)


            # new sample image, to fix the fact one patient has 10 images in average
            # change from 8 to 100 to check a distribution from a larger sample number.
            # torch.manual_seed(0)
            # self.image_index = torch.randint(high=self.hparams.test_batch_size, size=(8,))

            obs_batch_image = {k: v[self.image_index] for k, v in obs_batch.items()}
            self.log_img_grid('input', obs_batch_image['x'], save_img=False)

            if hasattr(self.pyro_model, 'reconstruct'):
                self.build_reconstruction(**obs_batch_image)


            conditions = {
                '40': {'age': torch.zeros_like(obs_batch['age']) + 40},
                '60': {'age': torch.zeros_like(obs_batch['age']) + 60},
                '80': {'age': torch.zeros_like(obs_batch['age']) + 80}
            }
            self.build_counterfactual('do(age=x)', obs=obs_batch, conditions=conditions, absolute2='T2D')
            self.build_counterfactual('do(age=x)', obs=obs_batch, conditions=conditions, absolute2='sex')

            conditions = {
                '0': {'sex': torch.zeros_like(obs_batch['sex'])},
                '1': {'sex': torch.ones_like(obs_batch['sex'])},
            }
            self.build_counterfactual('do(sex=x)', obs=obs_batch, conditions=conditions, absolute2='T2D')
            self.build_counterfactual('do(sex=x)', obs=obs_batch, conditions=conditions, absolute2='age')

            conditions = {
                '0': {'T2D': torch.zeros_like(obs_batch['T2D']).squeeze(-1)},
                '1': {'T2D': torch.ones_like(obs_batch['T2D']).squeeze(-1)},
            }
            self.build_counterfactual('do(T2D=x)', obs=obs_batch, conditions=conditions, absolute2='age')
            self.build_counterfactual('do(T2D=x)', obs=obs_batch, conditions=conditions, absolute2='sex')



    @classmethod
    def add_arguments(cls, parser):

        parser.add_argument('--data_dir', default="/mnt/alpha/diabetes/MS/data/images_original/", type=str, help="data train dir (default: %(default)s)")  # noqa: E501
        parser.add_argument('--split_dir', default="/mnt/beta/sliao/DSCM_fundus/deepscm/deepscm/datasets/dataset_selected_feature/", type=str, help="split dir (default: %(default)s)")  # noqa: E501

        parser.add_argument('--sample_img_interval', default=10, type=int, help="interval in which to sample and log images (default: %(default)s)")
        parser.add_argument('--train_batch_size', default=64, type=int, help="train batch size (default: %(default)s)")
        parser.add_argument('--test_batch_size', default=64, type=int, help="test batch size (default: %(default)s)")
        parser.add_argument('--validate', default= False, action='store_true', help="whether to validate (default: %(default)s)")
        parser.add_argument('--lr', default=1e-4, type=float, help="lr of deep part (default: %(default)s)")
        parser.add_argument('--pgm_lr', default=5e-3, type=float, help="lr of pgm (default: %(default)s)")
        parser.add_argument('--l2', default=0., type=float, help="weight decay (default: %(default)s)")
        parser.add_argument('--use_amsgrad', default=False, action='store_true', help="use amsgrad? (default: %(default)s)")
        parser.add_argument('--train_crop_type', default='random', choices=['random', 'center'], help="how to crop training images (default: %(default)s)")

        return parser
