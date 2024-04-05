# Verifying Quantum Circuits with Discrete-Time Barrier Certificates
Artifact for running checking properties of quantum circuits using adapted barrier certificate techniques.

## Running Examples
While within the root directory, run:

```./examples/example_<experiment>.sh```

This command runs multiple instances of the command:

```python -m examples.run_example -ex <experiment> --check -n <N> -k <inductive variable> --target <target state>```

To run individual experiments with different paramaters call the above line with your parameters set. For a full list of the parameters available and valid entries, run:

```python -m examples.run_example -h```