import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import optax

from NeuralNetworkEvolution.config import MLPConfig
from NeuralNetworkEvolution.activations import sin
from NeuralNetworkEvolution.mlp import CustomMLP, mlp_plot

import os
import sys
import logging

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

NUM_RUNS = 50

input_size = 1
hidden_sizes = [10, 10] 
output_size = 1
initial_activation_list = [sin]
activation_list = [sin]
bias = False
num_epochs = 10000
add_node_every = 50
threshold = 1e-4
n_samples = 2000
learning_rate = 0.01
start_seed = 0

config = MLPConfig(input_size=input_size,
                output_size=output_size,
                hidden_sizes=hidden_sizes,
                initial_activation_list=initial_activation_list,
                seed=start_seed)

config.__dict__.update({'n_samples': n_samples,
                        'learning_rate': learning_rate,
                        'num_epochs': num_epochs,
                        'add_node_every': add_node_every,
                        'threshold': threshold,
                        'activation_list': activation_list})

Description = f"Homo_poly_weight_strat__no_bias_{hidden_sizes[0]}_{hidden_sizes[1]}_{num_epochs}_{add_node_every}_{threshold}_runs_{NUM_RUNS}"
fig_folder = f"../figures/{Description}"
out_folder = f"../output/{Description}"
os.makedirs(fig_folder, exist_ok=True)
os.makedirs(out_folder, exist_ok=True)

logging.basicConfig(level=logging.INFO, filename=f"{out_folder}/info.log", filemode="w")
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)
logging.info(f"Description: {Description}")
logging.info(f"jax backend: {jax.lib.xla_bridge.get_backend().platform}")
logging.info(f"jax devices: {jax.devices()}")

def initialize_optimizer_state(mlp, optimizer):
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

def poly(x):
    return (x - 3)*(x - 2)*(x - 1)*x*(x + 1)*(x + 2)*(x + 3)

x = jnp.linspace(-3, 3, n_samples).reshape(-1, 1)
y = poly(x)

x_test = jnp.linspace(-3, 3, 100).reshape(-1, 1)
y_test = poly(x_test)

First_removal_history = []
threshold_history= []

for run in range(NUM_RUNS):
    logging.info(f"Run: {run}")
    run_output_folder = f"{out_folder}/run_{run}"
    os.makedirs(run_output_folder, exist_ok=True)

    config.seed = start_seed + run
    key = jax.random.PRNGKey(config.seed)
    key = jax.random.split(key, 1)[0]
    mlp = CustomMLP(config)
    opt = optax.adabelief(learning_rate=learning_rate)
    opt_state = initialize_optimizer_state(mlp, opt)

    initial_adjacency_matrix = mlp.adjacency_matrix()
    np.savetxt(f"{run_output_folder}/initial_adjacency_matrix.txt", initial_adjacency_matrix)

    Loss_history = []
    Node_history = []
    Update_history = []
    threshold_reached = False
    removed_neurons = False
    

    for epoch in range(num_epochs):
        loss, mlp, opt_state = train_step(mlp, x, y, opt_state, opt.update)

        key, add_key, sub_key = jax.random.split(key,3)
        n_neurons = sum(mlp.get_shape())
        logging.info(f"Epoch {epoch :03d}, Loss: {loss.item()}, Neurons: {n_neurons}")
        Loss_history.append(loss)
        Node_history.append(n_neurons)


        # Dynamically add or remove neurons
        if (epoch + 1) % add_node_every == 0:

            #add criterion
            if len(Update_history) == 0 or Update_history[-1][2] > loss or (Update_history[-1][3] == "removed" and Update_history[-2][3] == "removed" ):
                # if no previous addition or last addition was rejected, add a neuron
                # if last addition was accepted, add a neuron
                add_key, act_key = jax.random.split(add_key)
                activation = activation_list[jax.random.choice(key, jnp.arange(len(activation_list)))]
                layer = mlp.most_important_layer()
                mlp.add_neuron(layer_index=layer, activation=activation, bias = bias, key=add_key)
                opt_state = initialize_optimizer_state(mlp, opt)

                Update_history.append((epoch, n_neurons, loss, activation.__name__, layer))
                logging.info(f"Added neuron to hidden layer {layer+1} with activation {activation.__name__}")
                logging.info(f"network shape updated to :{mlp.get_shape()}")
            
            # remove criteria
            elif (Update_history[-1][3] == "removed" and Update_history[-2][2] < loss) or \
                (Update_history[-1][3] != "removed" and Update_history[-1][2] < loss):
                # if last addition was removed check loss against value before that
                # if last addition was accepted, check loss against it
                # if loss doesn't improve, reject it
                layer, neuron_idx = mlp.least_important_neuron()

                if len(mlp.layers[layer]) <= 1:
                    logging.info(f"Cannot remove neuron from layer {layer+1}, only one neuron left")
                    Update_history.append((epoch, n_neurons, loss, "single_node_layer", layer))
                    continue

                mlp.remove_neuron(layer_index=layer, neuron_index=neuron_idx)
                opt_state = initialize_optimizer_state(mlp, opt)
                Update_history.append((epoch, n_neurons, loss, "removed", layer))
                if not removed_neurons:
                    First_removal_history.append((epoch, n_neurons, loss))
                    removed_neurons = True
                    logging.info(f"First neuron removed at epoch {epoch} with network size {n_neurons} and loss {loss}")

                logging.info(f"Removed neuron to hidden layer {layer+1} at index {neuron_idx}")
                logging.info(f"network shape updated to :{mlp.get_shape()}")
            
            
        if loss < threshold:
            # if loss is below threshold, stop training
            threshold_reached = True
            logging.info(f"Threshold reached, stopping training at epoch {epoch}")
            threshold_history.append(epoch)
            break
    
    if not threshold_reached:
        logging.info(f"Threshold not reached, stopping training at epoch {epoch}")
        threshold_history.append(epoch)
    
    np.savetxt(f"{run_output_folder}/neurons.txt", Node_history)
    np.savetxt(f"{run_output_folder}/loss.txt", Loss_history)

    y_pred = jax.vmap(mlp)(x_test)
    np.savetxt(f"{run_output_folder}/y_pred.txt", y_pred)

    final_adjacency_matrix = mlp.adjacency_matrix()
    np.savetxt(f"{run_output_folder}/final_adjacency_matrix.txt", final_adjacency_matrix)
    final_shape = mlp.get_shape()
    np.savetxt(f"{run_output_folder}/final_shape.txt", final_shape)
    
    eqx.clear_caches()
    jax.clear_caches()

np.savetxt(f"{out_folder}/threshold_history.txt", threshold_history)
np.savetxt(f"{out_folder}/first_removal_history.txt", First_removal_history)