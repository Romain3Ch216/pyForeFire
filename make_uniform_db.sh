n_simulations=$1

for i in $(seq 1 $n_simulations);
do
    python test/make_uniform_simulation.py $i
    python test/uniform_simulation_db.py $i
done
