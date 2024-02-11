# NCA

Quantum impurity solver using the Non-Crossing Approximation (NCA).

## Installation

First clone the repo and `cd NCA`.
Then create a virtual python environment with `virtualenv venv`.
Get in the newly created virtual env with `. venv/bin/activate`.
Now, install the required libraries: `pip install -r requirements.txt`.
At that point you should be able to run the example: `python3 examples/steady_states.py`.

To setup a jupyter kernel with this virtual env, first install ipykernel (`pip install ipykernel`, and run it: `python -m ipykernel install --user --name=venv_NCA`.
You are ready to go.
