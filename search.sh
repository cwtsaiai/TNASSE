# TNASSE
python3 main.py --algo 'se' --runs 50 --max_evaluations 1000 --n 1 --h 3 --w 1 --sl 21 --ptype 'nasbench101' --atom 3 --score_function 'ni' --sigma 1
python3 main.py --algo 'se' --runs 50 --max_evaluations 1000 --n 1 --h 3 --w 1 --sl 6 --ptype 'nasbench201' --atom 5 --score_function 'ni' --sigma 1
python3 main.py --algo 'se' --runs 50 --max_evaluations 1000 --n 1 --h 3 --w 1 --sl 5 --ptype 'natsbenchsss' --atom 8 --score_function 'ni' --sigma 1

# NI
python3 main.py --algo 'rs' --runs 50 --max_evaluations 1000 --sl 21 --ptype 'nasbench101' --atom 3 --score_function 'ni' --sigma 1
python3 main.py --algo 'rs' --runs 50 --max_evaluations 1000 --sl 6 --ptype 'nasbench201' --atom 5 --score_function 'ni' --sigma 1
python3 main.py --algo 'rs' --runs 50 --max_evaluations 1000 --sl 5 --ptype 'natsbenchsss' --atom 8 --score_function 'ni' --sigma 1
