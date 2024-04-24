# Grover unmark examples
# Fails for longer timeout
# Tryout using only superposition state (or super w/ positive real+imag phases)
python -m examples.run_example -ex grover_dual --check -n 2 -k 1 --target 1 --mark 0 --runs 5
python -m examples.run_example -ex grover_dual --check -n 2 -k 2 --target 1 --mark 0 --runs 5
python -m examples.run_example -ex grover_odd_unmark --check -n 2 -k 1 --target 1 --mark 0 --runs 5
python -m examples.run_example -ex grover_even_unmark --check -n 2 -k 1 --target 1 --mark 0 --runs 5