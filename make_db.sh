n_simulations=$1

for i in $(seq 1 $n_simulations);
do
    python test/make_db.py $i
done
