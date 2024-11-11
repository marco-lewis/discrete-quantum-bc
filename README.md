# Verifying Quantum Circuits with Discrete-Time Barrier Certificates
Artifact for running checking properties of quantum circuits using adapted barrier certificate techniques.

## Installation
### Python Set-Up
Requires Python 3.10.15 and venv installed (may work with other Python versions but they are untested). Open a terminal in the repository directory (somewhere/discrete-quantum-bc) and run:

```python -m venv env```

Activate the environment:

```source env/bin/activate```

Install the requirements:

```pip install -r requirements.txt```

## dReal Set-Up
Follow the instructions to install dReal at https://github.com/dreal/dreal4

Once installed, either:
- Add the dReal directory to the repository directory
- Add the directory that the dReal binary is in to the $PATH.

## Running Examples
While within the repository directory, run:

```./examples/example_<experiment>.sh```

This command runs multiple instances of the command:

```python -m examples.run_example -ex <experiment> --check -n <N> -k <inductive variable> --target <target state>```

To run individual experiments with different paramaters call the above line with your parameters set. For a full list of the parameters available and valid entries, run:

```python -m examples.run_example -h```