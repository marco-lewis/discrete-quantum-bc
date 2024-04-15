# X gate examples
python -m examples.run_example -ex xgate --check -n 1 -k 2 --target 0 --runs 5 --gamma 0 --smt-timeout 300
python -m examples.run_example -ex xgate --check -n 1 -k 2 --target 1 --runs 5 --gamma 0 --smt-timeout 300
python -m examples.run_example -ex xgate --check -n 2 -k 2 --target 0 --runs 5 --gamma 0 --smt-timeout 300 --barrier-degree 2
python -m examples.run_example -ex xgate --check -n 2 -k 2 --target 1 --runs 5 --gamma 0 --smt-timeout 300 --barrier-degree 2
python -m examples.run_example -ex xgate --check -n 2 -k 2 --target 2 --runs 5 --gamma 0 --smt-timeout 300 --barrier-degree 2
python -m examples.run_example -ex xgate --check -n 2 -k 2 --target 3 --runs 5 --gamma 0 --smt-timeout 300 --barrier-degree 2