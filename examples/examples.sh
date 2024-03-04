# Z gate examples
python -m examples.run_example -ex zgate --check -n 1 -k 1 --target 0
python -m examples.run_example -ex zgate --check -n 1 -k 1 --target 1
python -m examples.run_example -ex zgate --check -n 2 -k 1 --target 0
# X gate examples
python -m examples.run_example -ex xgate --check -n 1 -k 1 --target 0
# python -m examples.run_example -ex xgate --check -n 1 -k 1 --target 1
# python -m examples.run_example -ex xgate --check -n 2 -k 1 --target 2
# XZ gate examples
python -m examples.run_example -ex xzgate --check -n 1 -k 2 --target 0
python -m examples.run_example -ex xzgate --check -n 1 -k 2 --target 1
python -m examples.run_example -ex xzgate --check -n 2 -k 2 --target 0
# Grover examples
python -m examples.run_example -ex grover_odd_unmark --check -n 2 -k 1 --target 3 --mark 2
python -m examples.run_example -ex grover_even_unmark --check -n 2 -k 1 --target 3 --mark 2
python -m examples.run_example -ex grover_odd_unmark --check -n 2 -k 1 --target 0 --mark 1
python -m examples.run_example -ex grover_even_unmark --check -n 2 -k 1 --target 0 --mark 1