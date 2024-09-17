n_simulations=$1

for i in $(seq 20 $n_simulations);
do
    python test/random_hill_simulation.py $i
    python test/hill_simulation_db.py $i
done
