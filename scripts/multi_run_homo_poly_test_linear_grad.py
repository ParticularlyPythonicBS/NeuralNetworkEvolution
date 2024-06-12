import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import optax
from sklearn.model_selection import train_test_split

from NeuralNetworkEvolution.config import MLPConfig
from NeuralNetworkEvolution.activations import sin
from NeuralNetworkEvolution.mlp import CustomMLP, mlp_plot

import os
import sys
import logging

# jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')

NUM_RUNS = 50

input_size = 1
hidden_sizes = [5, 5] 
output_size = 1
initial_activation_list = [sin]
activation_list = [sin]
bias = False
num_epochs = 25000
intervene_every = 100
start_seed = 0
threshold = 1e-4
grad_norm_threshold = 1e-3
n_samples = 20000
test_size = 0.2
learning_rate = 3e-4

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

Description = f"Homo_poly_test_linear_grad_strat__no_bias_{hidden_sizes[0]}_{hidden_sizes[1]}_{num_epochs}_{intervene_every}_{threshold}_runs_{NUM_RUNS}"
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

@eqx.filter_jit()
def test_step(mlp, x, y):
    return compute_loss(mlp, x, y)[0]

def grad_norm(grads):
    return jnp.sqrt(sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(grads)))

def poly(x):
    return (x - 3)*(x - 2)*(x - 1)*x*(x + 1)*(x + 2)*(x + 3)

x = jnp.linspace(-3, 3, n_samples).reshape(-1, 1)
y = poly(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=start_seed)

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
    init_neurons = sum(mlp.get_shape())
    opt = optax.adabelief(learning_rate=learning_rate)
    opt_state = initialize_optimizer_state(mlp, opt)

    initial_adjacency_matrix = mlp.adjacency_matrix()
    np.savetxt(f"{run_output_folder}/initial_adjacency_matrix.txt", initial_adjacency_matrix)

    train_loss_history = []
    test_loss_history = []
    node_history = []
    grad_norm_history = []
    graph_history = []
    update_history = []

    test_loss = np.inf # initialize test loss to infinity
    threshold_reached = False
    removed_neurons = False
    

    for epoch in range(num_epochs):
        train_loss, mlp, opt_state = train_step(mlp, x, y, opt_state, opt.update)
        _, grads  = compute_loss(mlp, x, y)
        grad_norm_val = grad_norm(grads)
        n_neurons = sum(mlp.get_shape())

        logging.info(f"Epoch {epoch :03d}, Loss: {train_loss.item()}, Neurons: {n_neurons}, Grad norm: {grad_norm_val :.3e}")
    
        train_loss_history.append((epoch, train_loss))
        grad_norm_history.append((epoch,grad_norm_val))
        node_history.append((epoch, n_neurons))

        if test_loss < threshold:
            # if loss is below threshold, stop training
            threshold_reached = True
            logging.info(f"Threshold reached, stopping training at epoch {epoch}")
            threshold_history.append(epoch)
            break

        if grad_norm_val < grad_norm_threshold/10: # stop training if gradient norm is very low
            logging.info(f"Gradient norm below threshold, stopping training at epoch {epoch}")
            break

        if ((epoch + 1) % intervene_every*int(n_neurons/init_neurons) == 0 # scale intervention period linearly with number of neurons
            or grad_norm_val < grad_norm_threshold) and epoch!=num_epochs-1: # intervene if gradient norm is below threshold, but not at last epoch
            test_loss = test_step(mlp, x_test, y_test)
            logging.info(f"Epoch {epoch :03d}, Test loss: {test_loss.item()}")
            test_loss_history.append((epoch,test_loss))

        # Neuron Addition criteria
        if (len(update_history) == 0    # if no previous addition
            or update_history[-1][3] > test_loss # if last addition was accepted
            or    (update_history[-1][-2] == "removed" and update_history[-2][-2] == "removed" )): # if two removals in a row

            add_key, act_key = jax.random.split(add_key)
            activation = activation_list[jax.random.choice(key, jnp.arange(len(activation_list)))]
            layers = len(mlp.get_shape()) - 1
            layer = jax.random.randint(act_key, (1,), 0, layers)[0] # randomly select a layer to add neuron to
            mlp.add_neuron(layer_index=layer, activation=activation, bias = bias, key=add_key)
            opt_state = initialize_optimizer_state(mlp, opt)

            update_history.append((epoch, n_neurons, train_loss, test_loss, activation.__name__, layer))
            logging.info(f"Added neuron to hidden layer {layer+1} with activation {activation.__name__}")
            logging.info(f"network shape updated to :{mlp.get_shape()}")
        
        # Neuron Removal criteria
        elif ((update_history[-1][-2] == "removed" and update_history[-2][3] < test_loss) # if last addition was removed check loss against value before that
            or (update_history[-1][-2] != "removed" and update_history[-1][3] < test_loss)): # if loss is worse than last accepted addition, reject it

            layer_key, neuron_key, sub_key = jax.random.split(sub_key,3)
            layer = update_history[-1][-1] # get the layer of last addition
            neuron_idx = len(mlp.layers[layer]) -1

            if len(mlp.layers[layer]) <= 1:
                logging.info(f"Cannot remove neuron from layer {layer+1}, only one neuron left")
                update_history.append((epoch, n_neurons, train_loss, test_loss, "single_node_layer", layer))
                continue

            mlp.remove_neuron(layer_index=layer, neuron_index=neuron_idx)
            opt_state = initialize_optimizer_state(mlp, opt)
            update_history.append((epoch, n_neurons, train_loss, test_loss, "removed", layer))
            
            if not removed_neurons:
                First_removal_history.append((epoch, n_neurons, train_loss, test_loss))
                removed_neurons = True
                logging.info(f"First neuron removed at epoch {epoch} with network size {n_neurons} and test loss {test_loss}")

            logging.info(f"Removed neuron to hidden layer {layer+1} at index {neuron_idx}")
            logging.info(f"network shape updated to :{mlp.get_shape()}")
    
    if not threshold_reached:
        logging.info(f"Threshold not reached, stopping training at epoch {epoch}")
        threshold_history.append(epoch)
    
    np.savetxt(f"{run_output_folder}/neurons.txt", node_history)
    np.savetxt(f"{run_output_folder}/train_loss.txt", train_loss_history)
    np.savetxt(f"{run_output_folder}/test_loss.txt", test_loss_history)

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