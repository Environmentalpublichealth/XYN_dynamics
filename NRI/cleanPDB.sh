gmx_mpi trjconv -s prod_2.tpr -f prod_final.xtc -o ca_trajectory.pdb -dt 100
# select 3 for alpha-carbon


awk '{if($0 ~ /^ATOM/) {print substr($0,1,21) " " substr($0,23)} else {print $0}}' ca_1.pdb > ca_clean.pdb

# split dataset
python convert_dataset.py --num-residues 178 --train-interval 40 --validate-interval 40 --test-interval 50