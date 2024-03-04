# Grover unmark examples
python -m examples.run_example -ex grover_odd_unmark --check -n 2 -k 1 --target 3 --mark 2
python -m examples.run_example -ex grover_even_unmark --check -n 2 -k 1 --target 3 --mark 2
python -m examples.run_example -ex grover_odd_unmark --check -n 2 -k 1 --target 0 --mark 1
python -m examples.run_example -ex grover_even_unmark --check -n 2 -k 1 --target 0 --mark 1