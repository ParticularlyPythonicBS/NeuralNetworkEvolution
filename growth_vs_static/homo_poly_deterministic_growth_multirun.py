import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import optax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from NeuralNetworkEvolution.config import MLPConfig
from NeuralNetworkEvolution.mlp import CustomMLP

import os

jax.config.update('jax_platform_name', 'cpu')

NUM_RUNS = 50

input_size = 1
hidden_sizes = [2, 2] 
min_neurons = 4
max_neurons = 32
output_size = 1
initial_activation_list = [jax.nn.tanh]
activation_list = [jax.nn.tanh]
optimizer = optax.adabelief
bias = False
num_epochs = 25000
intervene_every = 200
start_seed = 0
threshold = 1e-4
grad_norm_threshold = 1e-3
n_samples = 20000
test_size = 0.2
learning_rate = 3e-4

act_string = "_".join([act.__name__ for act in initial_activation_list])

config = MLPConfig(input_size=input_size,
                output_size=output_size,
                hidden_sizes=hidden_sizes,
                initial_activation_list=initial_activation_list,
                seed=start_seed)

config.__dict__.update({'n_samples': n_samples,
                        'learning_rate': learning_rate,
                        'num_epochs': num_epochs,
                        'intervene_every': intervene_every,
                        'threshold': threshold,
                        'activation_list': activation_list})

Description = f"Homo_{act_string}_poly_deterministic_addition_{optimizer.__name__}_no_bias_min_{min_neurons}_max_{max_neurons}_{hidden_sizes[0]}_{hidden_sizes[1]}_{num_epochs}_{intervene_every}_{start_seed}_{NUM_RUNS}"
fig_folder = f"../figures/multi_run/{Description}"
out_folder = f"../output/multi_run/{Description}"
os.makedirs(fig_folder, exist_ok=True)
os.makedirs(out_folder, exist_ok=True)

# data

def poly(x):
    """
    7th degree polynomial to predict
    """
    return (x - 3)*(x - 2)*(x - 1)*x*(x + 1)*(x + 2)*(x + 3)

x = jnp.linspace(-3, 3, n_samples).reshape(-1, 1)
y = poly(x)
scaler = MinMaxScaler(feature_range=(-1,1))
x_scaled, y_scaled = scaler.fit_transform(x), scaler.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=start_seed)

def initialize_optimizer_state(mlp, optimizer):
    """
    Optimizer initialization that filters for float arrays in the jax pytrees
    """
    return optimizer.init(eqx.filter(mlp, eqx.is_inexact_array))

@eqx.filter_value_and_grad()
def compute_loss(mlp, x, y):
    pred = jax.vmap(mlp)(x)
    return jnp.mean((pred - y) ** 2)

@eqx.filter_jit()
def train_step(mlp, x, y, opt_state, opt_update):
    loss, grads = compute_loss(mlp, x, y)
    updates, opt_state = opt_update(grads, opt_state)
    mlp = eqx.apply_updates(mlp, updates)
    return loss, mlp, opt_state

@eqx.filter_jit()
def test_step(mlp, x, y):
    return compute_loss(mlp, x, y)[0]

@eqx.filter_jit()
def grad_norm(grads):
    return jnp.sqrt(sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(grads)))

for run in range(NUM_RUNS):
    run_folder = f"{out_folder}/run_{run}"
    os.makedirs(run_folder, exist_ok=True)

    config.seed = start_seed +  run
    key = jax.random.PRNGKey(config.seed)
    key = jax.random.split(key, 1)[0]
    mlp = CustomMLP(config)
    init_neurons = sum(mlp.get_shape())
    opt = optimizer(learning_rate=learning_rate)
    opt_state = initialize_optimizer_state(mlp, opt)

    initial_adjacency_matrix = mlp.adjacency_matrix()
    np.savetxt(f"{run_folder}/initial_adjacency_matrix.txt", initial_adjacency_matrix)

    train_loss_history = []
    test_loss_history = []
    node_history = []
    grad_norm_history = []

    for epoch in range(num_epochs):
            train_loss, mlp, opt_state = train_step(mlp, x_train, y_train, opt_state, opt.update)
            _, grads  = compute_loss(mlp, x_train, y_train)
            grad_norm_val = grad_norm(grads)
            n_neurons = sum(mlp.get_shape())

            train_loss_history.append((epoch, train_loss))
            grad_norm_history.append((epoch,grad_norm_val))
            node_history.append((epoch, n_neurons))

            key, add_key, sub_key, prob_key = jax.random.split(key,4)


            if (epoch + 1) % intervene_every == 0 and epoch!=num_epochs-1: # intervene if gradient norm is below threshold, but not at last epoch
                test_loss = test_step(mlp, x_test, y_test)
                test_loss_history.append((epoch,test_loss))

                # Neuron Addition criteria
                if n_neurons < max_neurons:

                    add_key, act_key = jax.random.split(add_key)
                    activation = activation_list[jax.random.choice(key, jnp.arange(len(activation_list)))]
                    layers = len(mlp.get_shape()) - 1
                    layer = jax.random.randint(act_key, (1,), 0, layers)[0] # randomly select a layer to add neuron to
                    mlp.add_neuron(layer_index=layer, activation=activation, bias = bias, key=add_key)
                    opt_state = initialize_optimizer_state(mlp, opt)

        
    np.savetxt(f"{run_folder}/neurons.txt", node_history)
    np.savetxt(f"{run_folder}/train_loss.txt", train_loss_history)
    np.savetxt(f"{run_folder}/test_loss.txt", test_loss_history)

    y_pred = jax.vmap(mlp)(x_test)
    np.savetxt(f"{run_folder}/y_pred.txt", y_pred)

    final_adjacency_matrix = mlp.adjacency_matrix()
    np.savetxt(f"{run_folder}/final_adjacency_matrix.txt", final_adjacency_matrix)
    final_shape = mlp.get_shape()
    np.savetxt(f"{run_folder}/final_shape.txt", final_shape)
    
    eqx.clear_caches()
    jax.clear_caches()
