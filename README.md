# Freezing chaos without synaptic plasticity

# Overview
This repository provides the code for the manuscript “Freezing chaos without synaptic plasticity”.

Chaotic phenomena in RNN dynamics have been a focal point of research. In our model, we add an Onsager reaction term to the vanilla RNN dynamics. In this way, we find that the dynamical chaos is suppressed. We show that this freezing effect also holds in more biologically realistic networks, such as those composed of excitatory and inhibitory neurons.

# Requirement

To run the code in this repository, please ensure you have the following installed:

- **Python** (version 3.9 or above)
- **PyTorch** (version 1.12.0 or above, along with a compatible CUDA toolkit if GPU acceleration is desired)
- **NumPy** (version 1.21.0 or above)

It is recommended to use a virtual environment (e.g., with `conda` or `venv`) to manage the dependencies.

# parameters

1. N: neurons num
2. dt: discretized time step
3. step: number of simulation steps
4. gamma: regulatory factor
5. T: noise temperature

# Experiment results
You can get the experiment results in the following Python files.
1. "dynamics": dynamics.py
2. "Maximum Lyapunov exponent": MLE.py
3. "Jocabian matrix": Jocabian.py
4. "EI network": EI network.py and main.py
5. "Neural response latency": Neural response latency.py

Contact
If you have any question, please contact me via HuangWeizhong1999@163.com
