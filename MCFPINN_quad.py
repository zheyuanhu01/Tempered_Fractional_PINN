import haiku as hk
import jax
import optax
import jax.numpy as jnp
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
import os
import pandas as pd
from scipy.special import loggamma, gamma
from scipy.special import roots_jacobi

parser = argparse.ArgumentParser(description='PINN Training')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--dim', type=int, default=100) # dimension of the problem.
parser.add_argument('--epochs', type=int, default=100001) # Adam epochs
parser.add_argument('--lr', type=float, default=1e-3) # Adam lr
parser.add_argument('--PINN_h', type=int, default=128) # width of PINN
parser.add_argument('--PINN_L', type=int, default=4) # depth of PINN
parser.add_argument('--N_f', type=int, default=int(100)) # num of residual points
parser.add_argument('--N_mc', type=int, default=int(64)) # num of Monte Carlo points
parser.add_argument('--N_test', type=int, default=int(20000)) # num of test points
parser.add_argument('--save_loss', type=int, default=0) # flag for save loss or not

parser.add_argument('--r0', type=float, default=2) # r0 in MC-fPINN
parser.add_argument('--alpha', type=float, default=1.5) # tempered fractional alpha

parser.add_argument('--problem', type=int, default=7) # choose a problem with various exact sol

parser.add_argument('--unbiased', type=int, default=0) # biased/unbiased algorithm
args = parser.parse_args()
print(args)

from jax.config import config
config.update("jax_enable_x64", True)

np.random.seed(args.SEED)
key = jax.random.PRNGKey(args.SEED)

x = np.random.randn(args.N_test, args.dim)
r = np.random.rand(args.N_test, 1)
x = x / np.linalg.norm(x, axis=1, keepdims=True) * r
if args.problem == 1:
    u = (1 - np.sum(x ** 2, 1)) ** (1 + args.alpha / 2)
elif args.problem == 2:
    u = (1 - np.sum(x ** 2, 1)) ** (1 + args.alpha / 2) * x[:, 0]
elif args.problem == 3:
    coeff = np.random.randn(args.dim + 1)
    u = (1 - np.sum(x ** 2, 1)) ** (1 + args.alpha / 2) * (np.sum(coeff[:-1].reshape(1, -1) * x, 1) + coeff[-1])
elif args.problem == 4:
    u = (1 - np.sum(x ** 2, 1)) ** (args.alpha / 2)
elif args.problem == 5:
    u = (1 - np.sum(x ** 2, 1)) ** (args.alpha / 2) * x[:, 0]
elif args.problem == 6:
    coeff = np.random.randn(args.dim + 1)
    u = (1 - np.sum(x ** 2, 1)) ** (args.alpha / 2) * (np.sum(coeff[:-1].reshape(1, -1) * x, 1) + coeff[-1])
elif args.problem == 7:
    coeff1 = np.random.randn(args.dim + 1)
    u = (1 - np.sum(x ** 2, 1)) ** (args.alpha / 2) * (np.sum(coeff1[:-1].reshape(1, -1) * x, 1) + coeff1[-1])
    coeff2 = np.random.randn(args.dim + 1)
    u += (1 - np.sum(x ** 2, 1)) ** (1 + args.alpha / 2) * (np.sum(coeff2[:-1].reshape(1, -1) * x, 1) + coeff2[-1])
print(x.shape, u.shape)

dim = args.dim; alpha = args.alpha
log_S = np.log(2) + dim / 2 * np.log(np.pi) - loggamma(dim / 2)
log_C_d_alpha = alpha * np.log(2) + loggamma((alpha + dim) / 2) - dim / 2 * np.log(np.pi) - np.log(np.abs(gamma(-alpha / 2)))
log_const = log_S + log_C_d_alpha
const = np.exp(log_const)

const_res_1 = 0.5 # (args.r0 ** (2 - args.alpha)) / 2 / (2 - args.alpha)
const_res_2 = (args.r0 ** (- args.alpha)) / 2 / args.alpha

quad_x, quad_w = roots_jacobi(args.N_mc, 0, 1 - args.alpha)
#quad_x = jnp.clip(quad_x, a_min=args.eps)
# print(quad_x, quad_w)
    
# Adjust the nodes and weights to the interval [0, r0]
quad_x = args.r0 * (quad_x + 1) / 2
quad_w = quad_w * (args.r0 / 2) ** (2 - args.alpha)

class MLP(hk.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def __call__(self, x):
        boundary_aug = jax.nn.relu(1 - jnp.sum(x**2))
        # return jax.nn.relu(1 - jnp.sum(x ** 2)) ** (1 + args.alpha / 2) * (jnp.sum(coeff[:-1] * x) + coeff[-1])
        X = x
        for dim in self.layers[:-1]:
            X = hk.Linear(dim)(X)
            X = jnp.tanh(X)
        X = hk.Linear(self.layers[-1])(X)
        X = X[0]
        X = X * boundary_aug
        return X

class PINN:
    def __init__(self):
        self.epoch = args.epochs
        self.adam_lr = args.lr
        self.X, self.U = x, u
        self.quad_x, self.quad_w = quad_x, quad_w

        layers = [args.PINN_h] * (args.PINN_L - 1) + [1]
        @hk.transform
        def network(x):
            temp = MLP(layers=layers)
            return temp(x)
 
        self.u_net = hk.without_apply_rng(network)
        self.u_pred_fn = jax.vmap(self.u_net.apply, (None, 0)) # consistent with the dataset
        self.r1_pred_fn = jax.vmap(jax.vmap(self.residual1, (None, 0, None, None)), (None, None, 0, 0))
        self.r2_pred_fn = jax.vmap(self.residual2, (None, 0))

        self.params = self.u_net.init(key, self.X[0])
        lr = optax.linear_schedule(
            init_value=self.adam_lr, end_value=0,
            transition_steps=args.epochs,
            transition_begin=0
        )
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)

        self.saved_loss = []
        self.saved_l2 = []

    def resample(self, rng): # sample random points at the begining of each iteration
        keys = jax.random.split(rng, 4)
        N_f = args.N_f # Number of collocation points
        xf = jax.random.normal(keys[0], shape=(N_f, args.dim))
        rf = jax.random.uniform(keys[1], shape=(N_f, 1))
        xf = xf / jnp.linalg.norm(xf, axis=1, keepdims=True) * rf

        N_mc = args.N_mc # Number of Monte Carlo points
        xi = jax.random.normal(keys[2], shape=(N_mc, args.dim))
        xi = xi / jnp.linalg.norm(xi, axis=1, keepdims=True)

        if args.problem == 1:
            ff = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 2) + loggamma((args.alpha + args.dim) / 2) - loggamma(args.dim / 2)
            ff = np.exp(ff)
            ff = ff * (1 - (1 + args.alpha / args.dim) * jnp.sum(xf ** 2, 1))
        elif args.problem == 2:
            ff = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 2) + loggamma((args.alpha + args.dim) / 2 + 1) - loggamma(args.dim / 2 + 1)
            ff = np.exp(ff)
            ff = ff * (1 - (1 + args.alpha / (args.dim + 2)) * jnp.sum(xf ** 2, 1)) * xf[:, 0]
        elif args.problem == 3:
            ff = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 2) + loggamma((args.alpha + args.dim) / 2 + 1) - loggamma(args.dim / 2 + 1)
            ff = np.exp(ff)
            ff = ff * (1 - (1 + args.alpha / (args.dim + 2)) * jnp.sum(xf ** 2, 1)) * jnp.sum(coeff[:-1].reshape(1, -1) * xf, 1)

            ff2 = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 2) + loggamma((args.alpha + args.dim) / 2) - loggamma(args.dim / 2)
            ff2 = np.exp(ff2)
            ff2 = ff2 * (1 - (1 + args.alpha / args.dim) * jnp.sum(xf ** 2, 1))

            ff = ff + ff2 * coeff[-1]
        elif args.problem == 4:
            ff = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 1) + loggamma((args.alpha + args.dim) / 2) - loggamma(args.dim / 2)
            ff = np.exp(ff)
        elif args.problem == 5:
            ff = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 1) + loggamma((args.alpha + args.dim) / 2 + 1) - loggamma(args.dim / 2 + 1)
            ff = np.exp(ff)
            ff = ff * xf[:, 0]
        
        elif args.problem == 6:
            ff = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 1) + loggamma((args.alpha + args.dim) / 2) - loggamma(args.dim / 2)
            ff = np.exp(ff) * coeff[-1]

            ff2 = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 1) + loggamma((args.alpha + args.dim) / 2 + 1) - loggamma(args.dim / 2 + 1)
            ff2 = np.exp(ff2)
            ff2 = ff2 * jnp.sum(coeff[:-1].reshape(1, -1) * xf, 1)
            ff = ff + ff2

        elif args.problem == 7:
            ff = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 1) + loggamma((args.alpha + args.dim) / 2) - loggamma(args.dim / 2)
            ff = np.exp(ff) * coeff1[-1]

            ff2 = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 1) + loggamma((args.alpha + args.dim) / 2 + 1) - loggamma(args.dim / 2 + 1)
            ff2 = np.exp(ff2)
            ff2 = ff2 * jnp.sum(coeff1[:-1].reshape(1, -1) * xf, 1)
            ff_ = ff + ff2

            ff = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 2) + loggamma((args.alpha + args.dim) / 2 + 1) - loggamma(args.dim / 2 + 1)
            ff = np.exp(ff)
            ff = ff * (1 - (1 + args.alpha / (args.dim + 2)) * jnp.sum(xf ** 2, 1)) * jnp.sum(coeff2[:-1].reshape(1, -1) * xf, 1)

            ff2 = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 2) + loggamma((args.alpha + args.dim) / 2) - loggamma(args.dim / 2)
            ff2 = np.exp(ff2)
            ff2 = ff2 * (1 - (1 + args.alpha / args.dim) * jnp.sum(xf ** 2, 1))

            ff = ff + ff2 * coeff2[-1] + ff_

        return xf, xi, ff, keys[3]

    def residual1(self, params, x, xi, r):
        # print(x.shape, xi.shape, r.shape)
        u = self.u_net.apply(params, x)
        u_plus = self.u_net.apply(params, x + xi * r)
        u_minus = self.u_net.apply(params, x - xi * r)
        return const_res_1 * ((u - u_plus) / r  + (u - u_minus) / r) / r
        return const_res_1 * (2 * u - u_plus - u_minus) / r ** 2
    
    def residual2(self, params, x):
        # print(x.shape, xi.shape, r.shape)
        u = self.u_net.apply(params, x)
        #u_plus = self.u_net.apply(params, x + xi * r2)
        #u_minus = self.u_net.apply(params, x - xi * r2)
        return const_res_2 * (2 * u)

    def get_loss_pinn(self, params, xf, xi, ff):
        f1 = self.r1_pred_fn(params, xf, xi, self.quad_x)
        #print(f1.shape)
        f1 = (f1 * self.quad_w.reshape(-1, 1)).sum(0)
        f2 = self.r2_pred_fn(params, xf)
        f = (f1 + f2) * const
        #print(f.shape, ff.shape)
        mse_f = jnp.mean((f - ff)**2) #/ jnp.mean(ff**2)
        return mse_f

    @partial(jax.jit, static_argnums=(0,))
    def step_pinn(self, params, opt_state, rng):
        xf, xi, ff, rng = self.resample(rng)
        current_loss, gradients = jax.value_and_grad(self.get_loss_pinn)(params, xf, xi, ff)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, rng

    def train_adam(self):
        self.rng = jax.random.PRNGKey(args.SEED)
        for n in tqdm(range(self.epoch)):
            current_loss, self.params, self.opt_state, self.rng = self.step_pinn(self.params, self.opt_state, self.rng)
            if args.save_loss: current_l2 = self.L2_pinn(self.params, self.X, self.U)
            if n%1000==0: print('epoch %d, loss: %e, L2: %e'%(n, current_loss, self.L2_pinn(self.params, self.X, self.U)))
            if args.save_loss: self.saved_loss.append(current_loss)
            if args.save_loss: self.saved_l2.append(current_l2)
    @partial(jax.jit, static_argnums=(0,)) 
    def L2_pinn(self, params, x, u):
        pinn_u_pred_20 = self.u_pred_fn(params, x).reshape(-1)
        pinn_error_u_total_20 = jnp.linalg.norm(u - pinn_u_pred_20, 2) / jnp.linalg.norm(u, 2)
        return (pinn_error_u_total_20)

model = PINN()
model.train_adam()
# if not os.path.exists("records_hte"): os.makedirs("records_hte")
# if args.save_loss:
#     model.saved_loss = np.asarray(model.saved_loss)
#     model.saved_l2 = np.asarray(model.saved_l2)

#     info_dict = {"loss": model.saved_loss, "L2": model.saved_l2}
#     df = pd.DataFrame(data=info_dict, index=None)
#     df.to_excel(
#         "records_hte/Sin_Gordon_D="+str(args.dim)+"_method="+str(args.method)+"_"+args.algo+"_V="+str(args.V)+"_S="+str(args.SEED)+".xlsx",
#         index=False
#     )
