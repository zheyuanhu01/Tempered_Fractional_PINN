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
parser.add_argument('--dim', type=int, default=100) # dimension of the problem.
parser.add_argument('--epochs', type=int, default=10001) # Adam epochs
parser.add_argument('--lr', type=float, default=1e-3) # Adam lr
parser.add_argument('--PINN_h', type=int, default=128) # width of PINN
parser.add_argument('--PINN_L', type=int, default=4) # depth of PINN
parser.add_argument('--N_f', type=int, default=int(100)) # num of residual points
parser.add_argument('--N_mc', type=int, default=int(64)) # num of Monte Carlo points
parser.add_argument('--N_mc_test', type=int, default=int(1024)) # num of Monte Carlo points for label
parser.add_argument('--N_test', type=int, default=int(20000)) # num of test points
parser.add_argument('--save_loss', type=int, default=0) # flag for save loss or not

parser.add_argument('--Lambda', type=float, default=1) # tempered fractional lambda
parser.add_argument('--alpha', type=float, default=0.5) # tempered fractional alpha

parser.add_argument('--epsilon', type=float, default=1e-6) # truncation
args = parser.parse_args()
print(args)

jax.config.update("jax_enable_x64", True) # use float64 for numerical stability

np.random.seed(args.SEED)
key = jax.random.PRNGKey(args.SEED)

const_2 = 1
c = np.random.randn(1, args.dim - 1)
def generate_test_data(d):
    def func_u(x):
        temp =  1 - np.sum(x**2, 1)
        temp2 = c * np.sin(const_2 * (x[:, :-1] + np.cos(x[:, 1:]) + x[:, 1:] * np.cos(x[:, :-1])))
        temp2 = np.sum(temp2, 1)
        return temp * temp2
    N_test = args.N_test
    x = np.random.randn(N_test, d)
    r = np.random.rand(N_test, 1)
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * r
    u = func_u(x)
    return x, u
x, u = generate_test_data(d=args.dim)
print("Test data shape: ", x.shape, u.shape)

class MLP(hk.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def __call__(self, x):
        boundary_aug = jax.nn.relu(1 - jnp.sum(x**2))
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

        layers = [args.PINN_h] * (args.PINN_L - 1) + [1]
        @hk.transform
        def network(x):
            temp = MLP(layers=layers)
            return temp(x)
 
        self.u_net = hk.without_apply_rng(network)
        self.u_pred_fn = jax.vmap(self.u_net.apply, (None, 0)) # consistent with the dataset
        self.r_pred_fn = jax.vmap(jax.vmap(self.residual, (None, 0, None, None)), (None, None, 0, 0))
        self.r_exact_pred_fn = jax.vmap(jax.vmap(self.exact_residual, (0, None, None)), (None, 0, 0))

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

    def exact_solution(self, x):
        u1 = jax.nn.relu(1 - jnp.sum(x**2))
        x1, x2 = x[:-1], x[1:]
        coeffs = c.reshape(-1)
        u2 = coeffs * jnp.sin(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1)))
        u2 = jnp.sum(u2)
        u = (u1 * u2)
        return u

    def exact_residual(self, x, xi, r):
        # keys = jax.random.split(rng, 5)
        # N_mc = args.N_mc_test # Number of Monte Carlo points for label generetion
        # xi = jax.random.normal(keys[0], shape=(N_mc, args.dim))
        # xi = xi / jnp.linalg.norm(xi, axis=1, keepdims=True)

        # r = jax.random.gamma(keys[1], 2 - args.alpha, shape=(N_mc,)) / args.Lambda
        # r = jnp.clip(r, a_min=1e-3)
        u = self.exact_solution(x)
        u_plus = self.exact_solution(x + xi * r)
        u_minus = self.exact_solution(x - xi * r)
        return (2 * u - u_plus - u_minus) / r ** 2

    def resample(self, rng): # sample random points at the begining of each iteration
        keys = jax.random.split(rng, 7)
        N_f = args.N_f # Number of collocation points
        xf = jax.random.normal(keys[0], shape=(N_f, args.dim))
        rf = jax.random.uniform(keys[1], shape=(N_f, 1))
        xf = xf / jnp.linalg.norm(xf, axis=1, keepdims=True) * rf

        N_mc = args.N_mc # Number of Monte Carlo points
        xi = jax.random.normal(keys[2], shape=(N_mc, args.dim))
        xi = xi / jnp.linalg.norm(xi, axis=1, keepdims=True)
        r = jax.random.gamma(keys[3], 2 - args.alpha, shape=(N_mc,)) / args.Lambda
        r = jnp.clip(r, a_min=args.epsilon)

        N_mc = args.N_mc_test
        xi_test = jax.random.normal(keys[4], shape=(N_mc, args.dim))
        xi_test = xi_test / jnp.linalg.norm(xi_test, axis=1, keepdims=True)

        r_test = jax.random.gamma(keys[5], 2 - args.alpha, shape=(N_mc,)) / args.Lambda
        r_test = jnp.clip(r_test, a_min=args.epsilon)

        return xf, xi, r, xi_test, r_test, keys[6]

    def residual(self, params, x, xi, r):
        # print(x.shape, xi.shape, r.shape)
        u = self.u_net.apply(params, x)
        u_plus = self.u_net.apply(params, x + xi * r)
        u_minus = self.u_net.apply(params, x - xi * r)
        return (2 * u - u_plus - u_minus) / r ** 2

    def get_loss_pinn(self, params, xf, xi, r, ff):
        f = self.r_pred_fn(params, xf, xi, r)
        f = f.mean(0)
        #print(f.shape, ff.shape)
        mse_f = jnp.mean((f - ff)**2)
        return mse_f

    @partial(jax.jit, static_argnums=(0,))
    def step_pinn(self, params, opt_state, rng):
        xf, xi, r, xi_test, r_test, rng = self.resample(rng)
        ff = self.r_exact_pred_fn(xf, xi_test, r_test)
        #print(ff.shape)
        ff = ff.mean(0)
        #print(ff.shape)
        current_loss, gradients = jax.value_and_grad(self.get_loss_pinn)(params, xf, xi, r, ff)
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
