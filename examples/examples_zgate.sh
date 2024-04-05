# Z gate examples
python -m examples.run_example -ex zgate --check -n 1 -k 1 --target 0 --runs 5 --smt-timeout 1800 --barrier-degree 2
python -m examples.run_example -ex zgate --check -n 1 -k 1 --target 1 --runs 5 --smt-timeout 1800 --barrier-degree 2
python -m examples.run_example -ex zgate --check -n 2 -k 1 --target 0 --runs 5 --smt-timeout 1800 --barrier-degree 2
python -m examples.run_example -ex zgate --check -n 2 -k 1 --target 1 --runs 5 --smt-timeout 1800 --barrier-degree 2
python -m examples.run_example -ex zgate --check -n 2 -k 1 --target 2 --runs 5 --smt-timeout 1800 --barrier-degree 2
python -m examples.run_example -ex zgate --check -n 2 -k 1 --target 3 --runs 5 --smt-timeout 1800 --barrier-degree 2