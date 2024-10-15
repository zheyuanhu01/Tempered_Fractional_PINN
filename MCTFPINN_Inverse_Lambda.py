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

parser = argparse.ArgumentParser(description='PINN Training')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--dim', type=int, default=10) # dimension of the problem.
parser.add_argument('--epochs', type=int, default=100001) # Adam epochs
parser.add_argument('--lr', type=float, default=1e-3) # Adam lr
parser.add_argument('--PINN_h', type=int, default=128) # width of PINN
parser.add_argument('--PINN_L', type=int, default=4) # depth of PINN
parser.add_argument('--N_f', type=int, default=int(100)) # num of residual_lap points
parser.add_argument('--N_mc', type=int, default=int(128)) # num of Monte Carlo points
parser.add_argument('--N_mc_test', type=int, default=int(1024)) # num of Monte Carlo points for label

parser.add_argument('--lambda_x', type=float, default=2.0) # tempered fractional lambda 

parser.add_argument('--alpha', type=float, default=0.5) # tempered fractional alpha

parser.add_argument('--epsilon', type=float, default=1e-6) # truncation
args = parser.parse_args()
print(args)

jax.config.update("jax_enable_x64", True) # use float64 for numerical stability

np.random.seed(args.SEED)
key = jax.random.PRNGKey(args.SEED)

const_2 = 1
c = np.random.randn(1, args.dim - 1)

class MLP(hk.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def __call__(self, X):
        boundary_aug = jax.nn.relu(1 - jnp.sum(X**2))
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

        self.const_exact_lap = 0.5 * jax.scipy.special.gamma(2 - args.alpha) / (args.lambda_x ** (2 - args.alpha))

        layers = [args.PINN_h] * (args.PINN_L - 1) + [1]
        @hk.transform
        def network(x):
            temp = MLP(layers=layers)
            return temp(x)
 
        self.u_net = hk.without_apply_rng(network)
        self.u_pred_fn = jax.vmap(self.u_net.apply, (None, 0)) # consistent with the dataset

        self.r_lap_pred_fn = jax.vmap(jax.vmap(self.residual_lap, (None, 0, None, None, None, None)), (None, None, 0, 0, None, None))
        
        self.exact_pred_fn = jax.vmap(self.exact_solution, (0))
        self.lap_exact_pred_fn = jax.vmap(jax.vmap(self.exact_frac_lap, (0, None, None)), (None, 0, 0))

        self.params_net = self.u_net.init(key, jnp.zeros(args.dim))
        self.params = {
            'lambda': jnp.array(1.),
            'pinn': self.params_net
        }

        lr = optax.linear_schedule(
            init_value=self.adam_lr, end_value=0,
            transition_steps=args.epochs,
            transition_begin=0
        )
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)

        self.saved_loss = []
        self.saved_l2 = []

    def exact_solution(self, x):
        u1 = jax.nn.relu(1 - jnp.sum(x**2))
        x1, x2 = x[:-1], x[1:]
        coeffs = c.reshape(-1)
        u2 = coeffs * jnp.sin(x1 + jnp.cos(x2) + x2 * jnp.cos(x1))
        u2 = jnp.sum(u2)
        u = (u1 * u2)
        return u

    def exact_frac_lap(self, x, xi, r):
        u = self.exact_solution(x)
        u_plus = self.exact_solution(x + xi * r)
        u_minus = self.exact_solution(x - xi * r)
        return (2 * u - u_plus - u_minus) / r ** 2 * self.const_exact_lap

    def resample(self, rng): # sample random points at the begining of each iteration
        keys = jax.random.split(rng, 10)
        N_f = args.N_f # Number of collocation points
        xf = jax.random.normal(keys[0], shape=(N_f, args.dim))
        rf = jax.random.uniform(keys[1], shape=(N_f, 1))
        xf = xf / jnp.linalg.norm(xf, axis=1, keepdims=True) * rf
        uf = self.exact_pred_fn(xf)

        N_mc = args.N_mc # Number of Monte Carlo points
        xi = jax.random.normal(keys[3], shape=(N_mc, args.dim))
        xi = xi / jnp.linalg.norm(xi, axis=1, keepdims=True)
        r = jax.random.gamma(keys[4], 2 - args.alpha, shape=(N_mc,)) # / args.lambda_x
        r = jnp.clip(r, a_min=args.epsilon)

        N_mc = args.N_mc_test
        xi_test = jax.random.normal(keys[6], shape=(N_mc, args.dim))
        xi_test = xi_test / jnp.linalg.norm(xi_test, axis=1, keepdims=True)
        r_test = jax.random.gamma(keys[7], 2 - args.alpha, shape=(N_mc,)) / args.lambda_x
        r_test = jnp.clip(r_test, a_min=args.epsilon)

        return xf, uf, xi, r, xi_test, r_test, keys[9]

    def residual_lap(self, params, x, xi, r, alpha, lam):
        u = self.u_net.apply(params, x)
        u_plus = self.u_net.apply(params, x + xi * r / lam)
        u_minus = self.u_net.apply(params, x - xi * r / lam)
        const = 0.5 * (lam ** alpha) * jax.scipy.special.gamma(2 - args.alpha)
        return const * (2 * u - u_plus - u_minus) / r ** 2
    
    def get_loss_pinn(self, params_all, xf, uf, xi, r, ff):
        lam, params = params_all['lambda'], params_all['pinn']
        alpha = args.alpha
        # lam = args.lambda_x # jax.nn.sigmoid(lam) * 0.2 + 1
        
        data_loss = jnp.mean((self.u_pred_fn(params, xf) - uf) ** 2)

        f = self.r_lap_pred_fn(params, xf, xi, r, alpha, lam)
        f = f.mean(0)
        
        mse_f = jnp.mean((f - ff)**2)
        return mse_f + data_loss, (mse_f, data_loss)

    @partial(jax.jit, static_argnums=(0,))
    def step_pinn(self, params, opt_state, rng):
        xf, uf, xi, r, xi_test, r_test, rng = self.resample(rng)
        ff = self.lap_exact_pred_fn(xf, xi_test, r_test)
        ff = ff.mean(0)
        current_loss, gradients = jax.value_and_grad(self.get_loss_pinn, has_aux=True)(params, xf, uf, xi, r, ff)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, rng

    def train_adam(self):
        self.rng = jax.random.PRNGKey(args.SEED)
        for n in tqdm(range(self.epoch)):
            current_loss, self.params, self.opt_state, self.rng = self.step_pinn(self.params, self.opt_state, self.rng)
            lam = self.params['lambda']
            if n%1000==0: print('epoch %d, data loss: %e, fpinn loss: %e, lambda: %e'%(n, current_loss[1][1], current_loss[1][0], lam))

model = PINN()
model.train_adam()
