# Conventional molecular dynamics simulations
I followed the process described in [NRI-MD paper](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-29331-3/MediaObjects/41467_2022_29331_MOESM1_ESM.pdf)

## Prepare the PDB file for MD simulation
Remove H2O and add hydrogens
```bash
module load amber/22
reduce -BUILD < af3_model.pdb > wt_apo_ready.pdb 2> reduce_info.log
```
This outputs the PDB file for Amber to process

## Amber forcefield
Build a tleap.in file, include all the parameters amber required to calculate the forcefield. Input should be the xxx_ready.pdb produced from the reduce step.
```bash
tleap -f tleap.in > tleap.log 2>&1
```
This generates the following output files:
```txt
- wt_apo_ready_nonprot.pdb
- wt_apo_ready.pdb
- wt_apo_ready_renum.txt
- wt_apo_ready_sslink
- wt_apo_ready_water.pdb
- system.inpcrd
- system.prmtop
- tleap.log
```
Two system files are needed for the MD simulation run. 

## Amber to Gromacs
AmberTools needs GPU, to avoild the long queue for a GPU node, I converted amber outputs to use by GROMACS, a conventional MD tool on CPU. But I would still put amber scripts in here in case you can access a GPU. It should be 5x faster than GROMACS. 
### Amber run
Please find script and input files [here](https://github.com/Environmentalpublichealth/XYN_dynamics/tree/main/MDsimulation/AMBER).    
The bash script is `E7_amber.batch`. 

### GROMACS run
1. Convert Amber files into GROMACS inputs
```bash
 amber.python <<EOF
import parmed as pmd
try:
    print("Loading Amber files for protein")
    amber = pmd.load_file('system.prmtop', 'system.inpcrd')
    print("Writing GROMACS files...")
    amber.save('system.top', overwrite=True)
    amber.save('system.gro', overwrite=True)
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")
EOF
```
Note!! AmberTools needs to be loaded in the environment. I've try other ways to convert, only this script gave me gro and top files can be read by GROMACS. 

2. Build input mdp files
- min.mdp (minimization step)
- equi.mdp (equilibrium step)
- nvt.mdp (heat step)
- prod.mdp (simulation step)

All input files can be found in [here](https://github.com/Environmentalpublichealth/XYN_dynamics/tree/main/MDsimulation/GROMACS).     
The bash scripts are `WT_GMX.batch` and `GMX_con.batch`.

The MD simulations on CPU runs slow, at about 30 - 50 ns per day. If we need to run a 200 ns simulation, it will span 4 to 7 days. My script has a initial GMX start file and a continue file to pick up the leftover ns every 48 hours because the max time I can submit a batch job is 2 days. If no CPU time use limit, no need to do the continue script. 

## Post-analysis
After the simulation finished, output files generated:
```txt
- prod_1.cpt
- prod_1.edr
- prod_1.gro
- prod_1.tpr
- prod_1.xtc
- prod_1.log
```
Script to generate RMSF and RMSD for visualization. `run_analysis.sh`
