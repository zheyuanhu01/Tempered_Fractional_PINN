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
parser.add_argument('--N_mc', type=int, default=int(64)) # num of Monte Carlo points
parser.add_argument('--N_mc_test', type=int, default=int(1024)) # num of Monte Carlo points for label
parser.add_argument('--N_test', type=int, default=int(20000)) # num of test points
parser.add_argument('--save_loss', type=int, default=0) # flag for save loss or not

parser.add_argument('--lambda_x', type=float, default=1) # tempered fractional lambda
parser.add_argument('--lambda_t', type=float, default=1) # tempered fractional time lambda
parser.add_argument('--alpha', type=float, default=0.5) # tempered fractional alpha
parser.add_argument('--gamma', type=float, default=0.5) # tempered fractional gamma

parser.add_argument('--epsilon', type=float, default=1e-6) # truncation
args = parser.parse_args()
print(args)

jax.config.update("jax_enable_x64", True) # use float64 for numerical stability

np.random.seed(args.SEED)
key = jax.random.PRNGKey(args.SEED)

const_2 = 1
c = np.random.randn(1, args.dim - 1)
v = np.random.rand(args.dim)
def generate_test_data(d):
    def func_u(x, t):
        temp =  1 - np.sum(x**2, 1)
        temp2 = c * np.sin(t.reshape(-1, 1) * (x[:, :-1] + np.cos(x[:, 1:]) + x[:, 1:] * np.cos(x[:, :-1])))
        temp2 = np.sum(temp2, 1)
        return temp * temp2
    N_test = args.N_test
    x = np.random.randn(N_test, d)
    r = np.random.rand(N_test, 1)
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * r
    t = np.random.rand(N_test)
    u = func_u(x, t)
    return x, t, u
x, t, u = generate_test_data(d=args.dim)
print("Test data shape: ", x.shape, t.shape, u.shape)

class MLP(hk.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def __call__(self, x, t):
        boundary_aug = jax.nn.relu(1 - jnp.sum(x**2)) * t
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

        layers = [args.PINN_h] * (args.PINN_L - 1) + [1]
        @hk.transform
        def network(x, t):
            temp = MLP(layers=layers)
            return temp(x, t)
 
        self.u_net = hk.without_apply_rng(network)
        self.u_pred_fn = jax.vmap(self.u_net.apply, (None, 0, 0)) # consistent with the dataset

        self.r_lap_pred_fn = jax.vmap(jax.vmap(self.residual_lap, (None, 0, 0, None, None)), (None, None, None, 0, 0))
        self.r_t1_pred_fn = jax.vmap(jax.vmap(self.residual_time_frac1, (None, 0, 0, None)), (None, None, None, 0))
        self.r_t2_pred_fn = jax.vmap(self.residual_time_frac2, (None, 0, 0, 0))
        self.r_v_pred_fn = jax.vmap(self.residual_v, (None, 0, 0))

        self.lap_exact_pred_fn = jax.vmap(jax.vmap(self.exact_frac_lap, (0, 0, None, None)), (None, None, 0, 0))
        self.t1_exact_pred_fn = jax.vmap(jax.vmap(self.exact_time_frac1, (0, 0, None)), (None, None, 0))
        self.t2_exact_pred_fn = jax.vmap(self.exact_time_frac2, (0, 0, 0))
        self.v_exact_pred_fn = jax.vmap(self.exact_v, (0, 0))

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

    def exact_solution(self, x, t):
        u1 = jax.nn.relu(1 - jnp.sum(x**2))
        x1, x2 = x[:-1], x[1:]
        coeffs = c.reshape(-1)
        u2 = coeffs * jnp.sin(t * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1)))
        u2 = jnp.sum(u2)
        u = (u1 * u2)
        return u

    def exact_frac_lap(self, x, t, xi, r):
        u = self.exact_solution(x, t)
        u_plus = self.exact_solution(x + xi * r, t)
        u_minus = self.exact_solution(x - xi * r, t)
        return (2 * u - u_plus - u_minus) / r ** 2
    
    def exact_time_frac1(self, x, t, tau):
        u = self.exact_solution(x, t)
        u0 = self.exact_solution(x, t - tau * t)
        return args.gamma / (1 - args.gamma) * (t ** (1 - args.gamma)) * (u - u0) / tau / t
    
    def exact_time_frac2(self, x, t, t0):
        u = self.exact_solution(x, t)
        u0 = self.exact_solution(x, t0)
        return (u - u0) / (t ** args.gamma)

    def exact_v(self, x, t):
        fn = lambda x: self.exact_solution(x, t)
        return jax.jvp(fn, (x, ), (v, ))[1]

    def resample(self, rng): # sample random points at the begining of each iteration
        keys = jax.random.split(rng, 10)
        N_f = args.N_f # Number of collocation points
        xf = jax.random.normal(keys[0], shape=(N_f, args.dim))
        rf = jax.random.uniform(keys[1], shape=(N_f, 1))
        xf = xf / jnp.linalg.norm(xf, axis=1, keepdims=True) * rf
        tf = jax.random.uniform(keys[2], shape=(N_f,))

        N_mc = args.N_mc # Number of Monte Carlo points
        xi = jax.random.normal(keys[3], shape=(N_mc, args.dim))
        xi = xi / jnp.linalg.norm(xi, axis=1, keepdims=True)
        r = jax.random.gamma(keys[4], 2 - args.alpha, shape=(N_mc,)) / args.lambda_x
        r = jnp.clip(r, a_min=args.epsilon)
        t0 = jnp.zeros((N_f, ))
        tau = jax.random.beta(keys[5], 1 - args.gamma, 1, shape=(N_mc,))

        N_mc = args.N_mc_test
        xi_test = jax.random.normal(keys[6], shape=(N_mc, args.dim))
        xi_test = xi_test / jnp.linalg.norm(xi_test, axis=1, keepdims=True)
        r_test = jax.random.gamma(keys[7], 2 - args.alpha, shape=(N_mc,)) / args.lambda_x
        r_test = jnp.clip(r_test, a_min=args.epsilon)
        t0_test = jnp.zeros((N_f, ))
        tau_test = jax.random.beta(keys[8], 1 - args.gamma, 1, shape=(N_mc,))

        return xf, tf, xi, r, t0, tau, xi_test, r_test, t0_test, tau_test, keys[9]

    def residual_lap(self, params, x, t, xi, r):
        u = self.u_net.apply(params, x, t)
        u_plus = self.u_net.apply(params, x + xi * r, t)
        u_minus = self.u_net.apply(params, x - xi * r, t)
        return (2 * u - u_plus - u_minus) / r ** 2
    
    def residual_time_frac1(self, params, x, t, tau):
        u = self.u_net.apply(params, x, t)
        u0 = self.u_net.apply(params, x, t - tau * t)
        return args.gamma / (1 - args.gamma) * (t ** (1 - args.gamma)) * (u - u0) / tau / t
    
    def residual_time_frac2(self, params, x, t, t0):
        u = self.u_net.apply(params, x, t)
        u0 = self.u_net.apply(params, x, t0)
        return (u - u0) / (t ** args.gamma)
    
    def residual_v(self, params, x, t):
        fn = lambda x: self.u_net.apply(params, x, t)
        return jax.jvp(fn, (x, ), (v, ))[1]

    def get_loss_pinn(self, params, xf, tf, xi, r, t0, tau, ff):
        f = self.r_lap_pred_fn(params, xf, tf, xi, r)
        f = f.mean(0)

        ft1 = self.r_t1_pred_fn(params, xf, tf, tau)
        ft1 = ft1.mean(0)

        ft2 = self.r_t2_pred_fn(params, xf, tf, t0)

        fv = self.r_v_pred_fn(params, xf, tf)

        print(f.shape, ft1.shape, ft2.shape, fv.shape)
        
        mse_f = jnp.mean((f + ft1 + ft2 + fv - ff)**2)
        return mse_f

    @partial(jax.jit, static_argnums=(0,))
    def step_pinn(self, params, opt_state, rng):
        xf, tf, xi, r, t0, tau, xi_test, r_test, t0_test, tau_test, rng = self.resample(rng)
        ff = self.lap_exact_pred_fn(xf, tf, xi_test, r_test)
        ff_t1 = self.t1_exact_pred_fn(xf, tf, tau_test)
        ff_t2 = self.t2_exact_pred_fn(xf, tf, t0_test)
        ff_v = self.v_exact_pred_fn(xf, tf)
        print(ff.shape, ff_t1.shape, ff_t2.shape, ff_v.shape)
        ff = ff.mean(0)
        ff_t1 = ff_t1.mean(0)
        ff = ff + ff_t1 + ff_t2 + ff_v
        current_loss, gradients = jax.value_and_grad(self.get_loss_pinn)(params, xf, tf, xi, r, t0, tau, ff)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, rng

    def train_adam(self):
        self.rng = jax.random.PRNGKey(args.SEED)
        for n in tqdm(range(self.epoch)):
            current_loss, self.params, self.opt_state, self.rng = self.step_pinn(self.params, self.opt_state, self.rng)
            if args.save_loss: current_l2 = self.L2_pinn(self.params, self.X, self.U)
            if n%1000==0: print('epoch %d, loss: %e, L2: %e'%(n, current_loss, self.L2_pinn(self.params, self.X, self.T, self.U)))
            if args.save_loss: self.saved_loss.append(current_loss)
            if args.save_loss: self.saved_l2.append(current_l2)
    @partial(jax.jit, static_argnums=(0,)) 
    def L2_pinn(self, params, x, t, u):
        pinn_u_pred_20 = self.u_pred_fn(params, x, t).reshape(-1)
        pinn_error_u_total_20 = jnp.linalg.norm(u - pinn_u_pred_20, 2) / jnp.linalg.norm(u, 2)
        return (pinn_error_u_total_20)

model = PINN()
model.train_adam()
