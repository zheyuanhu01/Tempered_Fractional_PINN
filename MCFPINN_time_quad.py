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
parser.add_argument('--SEED', type=int, default=0) # Random seed
parser.add_argument('--dim', type=int, default=10) # dimension of the problem
parser.add_argument('--epochs', type=int, default=100001) # Adam epochs
parser.add_argument('--lr', type=float, default=1e-3) # Adam lr
parser.add_argument('--PINN_h', type=int, default=128) # width of PINN
parser.add_argument('--PINN_L', type=int, default=4) # depth of PINN
parser.add_argument('--N_f', type=int, default=int(100)) # num of residual points
parser.add_argument('--N_mc', type=int, default=int(64)) # num of Monte Carlo points for fractional Lap
parser.add_argument('--N_test', type=int, default=int(20000)) # num of test points
parser.add_argument('--save_loss', type=int, default=0) # flag for save loss or not

parser.add_argument('--r0', type=float, default=2) # r0 in MC-fPINN
parser.add_argument('--eps', type=float, default=1e-6) # epsilon in truncating MC-fPINN
parser.add_argument('--alpha', type=float, default=1.5) # fractional alpha
parser.add_argument('--gamma', type=float, default=0.5) # time fractinal gamma
parser.add_argument('--c', type=float, default=1) # diffusion c

parser.add_argument('--T', type=float, default=1) # terminal time
parser.add_argument('--unbiased', type=int, default=0) # biased/unbiased algorithm
parser.add_argument('--problem', type=int, default=0) # choose a problem

args = parser.parse_args()
print(args)

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True) # use float64 for numerical stability

np.random.seed(args.SEED)
key = jax.random.PRNGKey(args.SEED)

t = np.random.rand(args.N_test) * args.T
x = np.random.randn(args.N_test, args.dim)
r = np.random.rand(args.N_test, 1)
x = x / np.linalg.norm(x, axis=1, keepdims=True) * r

v = np.zeros(args.dim)
v = np.random.rand(args.dim)

if args.problem == 0:
    coeff2 = np.random.randn(args.dim + 1)
    u = (1 - np.sum(x ** 2, 1)) ** (1 + args.alpha / 2) * (np.sum(coeff2[:-1].reshape(1, -1) * x, 1) + coeff2[-1])
    u = t * u
elif args.problem == 1:
    coeff1 = np.random.randn(args.dim + 1)
    u = (1 - np.sum(x ** 2, 1)) ** (args.alpha / 2) * (np.sum(coeff1[:-1].reshape(1, -1) * x, 1) + coeff1[-1])
    coeff2 = np.random.randn(args.dim + 1)
    u += (1 - np.sum(x ** 2, 1)) ** (1 + args.alpha / 2) * (np.sum(coeff2[:-1].reshape(1, -1) * x, 1) + coeff2[-1])
    u = t * u
elif args.problem == 2:
    u = t * (1 - np.sum(x ** 2, 1)) ** (1 + args.alpha / 2)
print(x.shape, u.shape)

dim = args.dim; alpha = args.alpha
log_S = np.log(2) + dim / 2 * np.log(np.pi) - loggamma(dim / 2)
log_C_d_alpha = alpha * np.log(2) + loggamma((alpha + dim) / 2) - dim / 2 * np.log(np.pi) - np.log(np.abs(gamma(-alpha / 2)))
log_const = log_S + log_C_d_alpha
const = np.exp(log_const)

const_res_1 = 0.5 # (args.r0 ** (2 - args.alpha)) / 2 / (2 - args.alpha)
const_res_2 = (args.r0 ** (- args.alpha)) / 2 / args.alpha

quad_x, quad_w = roots_jacobi(args.N_mc, 0, 1 - args.alpha)

quad_t, quad_wt = roots_jacobi(args.N_mc, 0, -args.gamma)
quad_t = (quad_t + 1) / 2
quad_wt = quad_wt * (1 / 2) ** (1 - args.gamma)

class MLP(hk.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def __call__(self, x, t):
        # return t * (jax.nn.relu(1 - jnp.sum(x ** 2)) ** (1 + args.alpha / 2) * (jnp.sum(coeff2[:-1] * x) + coeff2[-1]))
        # return t * (jax.nn.relu(1 - jnp.sum(x ** 2)) ** (args.alpha / 2) * (jnp.sum(coeff1[:-1] * x) + coeff1[-1]) + jax.nn.relu(1 - jnp.sum(x ** 2)) ** (1 + args.alpha / 2) * (jnp.sum(coeff2[:-1] * x) + coeff2[-1]))
        boundary_aug = jax.nn.relu(1 - jnp.sum(x ** 2)) * t
        X = jnp.hstack([x, t])
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
        self.X, self.T, self.U = x, t, u
        self.quad_x, self.quad_w = quad_x, quad_w
        self.quad_t, self.quad_wt = quad_t, quad_wt

        layers = [args.PINN_h] * (args.PINN_L - 1) + [1]
        @hk.transform
        def network(x, t):
            temp = MLP(layers=layers)
            return temp(x, t)
 
        self.u_net = hk.without_apply_rng(network)
        self.u_pred_fn = jax.vmap(self.u_net.apply, (None, 0, 0)) # consistent with the dataset
        self.r1_pred_fn = jax.vmap(jax.vmap(self.residual1, (None, 0, 0, None, None)), (None, None, None, 0, 0))
        self.r2_pred_fn = jax.vmap(self.residual2, (None, 0, 0))
        self.r3_pred_fn = jax.vmap(self.residual3, (None, 0, 0))

        self.t1_pred_fn = jax.vmap(self.time_residual1, (None, 0, 0, 0))
        self.t2_pred_fn = jax.vmap(jax.vmap(self.time_residual2, (None, 0, 0, None)), (None, None, None, 0))

        self.params = self.u_net.init(key, self.X[0], self.T[0])
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
        keys = jax.random.split(rng, 8)
        N_f = args.N_f # Number of collocation points
        xf = jax.random.normal(keys[0], shape=(N_f, args.dim))
        rf = jax.random.uniform(keys[1], shape=(N_f, 1)) * 1
        xf = xf / jnp.linalg.norm(xf, axis=1, keepdims=True) * rf

        N_mc = args.N_mc # Number of Monte Carlo points
        xi = jax.random.normal(keys[2], shape=(N_mc, args.dim))
        xi = xi / jnp.linalg.norm(xi, axis=1, keepdims=True)

        tf = jax.random.uniform(keys[5], shape=(N_f, )) * args.T
        t0 = jnp.zeros((N_f, ))
        if args.problem == 2:
            ff = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 2) + loggamma((args.alpha + args.dim) / 2) - loggamma(args.dim / 2)
            ff = np.exp(ff)
            ff = ff * (1 - (1 + args.alpha / args.dim) * jnp.sum(xf ** 2, 1))
            ff = tf * ff

            ff2 = tf ** (1 - args.gamma) / (1 - args.gamma) * (1 - jnp.sum(xf ** 2, 1)) ** (1 + args.alpha / 2)

            ff3 = -(1 + args.alpha / 2) * (1 - np.sum(xf ** 2, 1)) ** (args.alpha / 2) * 2 * np.sum(xf * v.reshape(1, -1), 1)
            ff3 *= tf
        elif args.problem == 1:
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

            ff = tf * ff

            ff2 = (1 - jnp.sum(xf ** 2, 1)) ** (args.alpha / 2) * (jnp.sum(coeff1[:-1].reshape(1, -1) * xf, 1) + coeff1[-1])
            ff2 += (1 - jnp.sum(xf ** 2, 1)) ** (1 + args.alpha / 2) * (jnp.sum(coeff2[:-1].reshape(1, -1) * xf, 1) + coeff2[-1])
            ff2 = tf ** (1 - args.gamma) / (1 - args.gamma) * ff2

            ff3 = (1 - np.sum(xf ** 2, 1)) ** (1 + args.alpha / 2) * np.sum(coeff2[:-1] * v)
            ff3 -= (1 + args.alpha / 2) * (1 - np.sum(xf ** 2, 1)) ** (args.alpha / 2) * 2 * np.sum(xf * v.reshape(1, -1), 1) * (np.sum(coeff2[:-1].reshape(1, -1) * xf, 1) + coeff2[-1])
            ff3 += (1 - np.sum(xf ** 2, 1)) ** (args.alpha / 2) * np.sum(coeff1[:-1] * v)
            ff3 -= (args.alpha / 2) * (1 - np.sum(xf ** 2, 1)) ** (args.alpha / 2 - 1) * 2 * np.sum(xf * v.reshape(1, -1), 1) * (np.sum(coeff1[:-1].reshape(1, -1) * xf, 1) + coeff1[-1])
            ff3 *= tf
        elif args.problem == 0:
            ff = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 2) + loggamma((args.alpha + args.dim) / 2 + 1) - loggamma(args.dim / 2 + 1)
            ff = np.exp(ff)
            ff = ff * (1 - (1 + args.alpha / (args.dim + 2)) * jnp.sum(xf ** 2, 1)) * jnp.sum(coeff2[:-1].reshape(1, -1) * xf, 1)
            ff2 = args.alpha * np.log(2) + loggamma(args.alpha / 2 + 2) + loggamma((args.alpha + args.dim) / 2) - loggamma(args.dim / 2)
            ff2 = np.exp(ff2)
            ff2 = ff2 * (1 - (1 + args.alpha / args.dim) * jnp.sum(xf ** 2, 1))
            ff = ff + ff2 * coeff2[-1]
            ff = tf * ff

            ff2 = (1 - jnp.sum(xf ** 2, 1)) ** (1 + args.alpha / 2) * (jnp.sum(coeff2[:-1].reshape(1, -1) * xf, 1) + coeff2[-1])
            ff2 = tf ** (1 - args.gamma) / (1 - args.gamma) * ff2

            # ff3 = (1 - np.sum(x ** 2, 1)) ** (1 + args.alpha / 2) * (np.sum(coeff2[:-1].reshape(1, -1) * x, 1) + coeff2[-1])
            ff3 = (1 - np.sum(xf ** 2, 1)) ** (1 + args.alpha / 2) * np.sum(coeff2[:-1] * v)
            ff3 -= (1 + args.alpha / 2) * (1 - np.sum(xf ** 2, 1)) ** (args.alpha / 2) * 2 * np.sum(xf * v.reshape(1, -1), 1) * (np.sum(coeff2[:-1].reshape(1, -1) * xf, 1) + coeff2[-1])
            ff3 *= tf
        return xf, tf, t0, xi, args.c * ff + ff2 + ff3, keys[7]
    
    def time_residual1(self, params, x, t, t0):
        u = self.u_net.apply(params, x, t)
        u0 = self.u_net.apply(params, x, t0)
        return (u - u0) / (t ** args.gamma)
    
    def time_residual2(self, params, x, t, tau):
        u = self.u_net.apply(params, x, t)
        u0 = self.u_net.apply(params, x, t - tau * t)
        return args.gamma * (t ** (1 - args.gamma)) * (u - u0) / tau / t

    def residual1(self, params, x, t, xi, r):
        u = self.u_net.apply(params, x, t)
        u_plus = self.u_net.apply(params, x + xi * r, t)
        u_minus = self.u_net.apply(params, x - xi * r, t)
        return const_res_1 * (2 * u - u_plus - u_minus) / r ** 2
    
    def residual2(self, params, x, t):
        u = self.u_net.apply(params, x, t)
        return const_res_2 * (2 * u)
    
    def residual3(self, params, x, t):
        fn = lambda x: self.u_net.apply(params, x, t)
        return jax.jvp(fn, (x, ), (v, ))[1]

    def get_loss_pinn(self, params, xf, tf, t0, xi, ff):
        f1 = self.r1_pred_fn(params, xf, tf, xi, self.quad_x)
        f1 = (f1 * self.quad_w.reshape(-1, 1)).sum(0)
        f2 = self.r2_pred_fn(params, xf, tf)
        f2 = f2.mean(0)
        f = (f1 + f2) * const

        f3 = self.r3_pred_fn(params, xf, tf)
        f += f3

        t1 = self.t1_pred_fn(params, xf, tf, t0)
        t2 = self.t2_pred_fn(params, xf, tf, self.quad_t)
        # print(t1.shape, t2.shape, f.shape, ff.shape)
        t2 = t2 * self.quad_wt.reshape(-1, 1)
        t2 = t2.sum(0)
        tt = t1 + t2
        mse_f = jnp.mean((tt + f - ff)**2)
        return mse_f

    @partial(jax.jit, static_argnums=(0,))
    def step_biased(self, params, opt_state, rng):
        xf, tf, t0, xi, ff, rng = self.resample(rng)
        current_loss, gradients = jax.value_and_grad(self.get_loss_pinn)(params, xf, tf, t0, xi, ff)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, rng

    def train_adam(self):
        self.rng = jax.random.PRNGKey(args.SEED)
        for n in tqdm(range(self.epoch)):
            current_loss, self.params, self.opt_state, self.rng = self.step_biased(self.params, self.opt_state, self.rng)
            if args.save_loss: current_l2 = self.L2_pinn(self.params, self.X, self.T, self.U)
            if n%1000==0: print('epoch %d, loss: %e, L2: %e'%(n, current_loss, self.L2_pinn(self.params, self.X, self.T, self.U)))
            if args.save_loss: self.saved_loss.append(current_loss); self.saved_l2.append(current_l2)
    @partial(jax.jit, static_argnums=(0,)) 
    def L2_pinn(self, params, x, t, u):
        pinn_u_pred_20 = self.u_pred_fn(params, x, t).reshape(-1)
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
