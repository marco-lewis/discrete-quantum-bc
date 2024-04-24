# Grover unmark examples
python -m examples.run_example -ex grover_dual --check -n 2 -k 1 --target 1 --mark 0 --runs 5
python -m examples.run_example -ex grover_dual --check -n 2 -k 2 --target 1 --mark 0 --runs 5
python -m examples.run_example -ex grover_odd_unmark --check -n 2 -k 1 --target 1 --mark 0 --runs 5
python -m examples.run_example -ex grover_even_unmark --check -n 2 -k 1 --target 1 --mark 0 --runs 5