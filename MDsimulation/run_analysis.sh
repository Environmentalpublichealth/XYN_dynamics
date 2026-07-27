#!/bin/bash
#SBATCH --job-name=Post_Analysis
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4      # 4 cores is plenty for post-processing
#SBATCH --mem=16G
#SBATCH --time=01:00:00        # Analysis is fast, 2 hours is more than enough
#SBATCH --output=analysis_out_%j.log
#SBATCH --error=analysis_err_%j.log

# 1. Clean environment and load GROMACS
module purge
module load gromacs/2022.5_plumed_gcc_9.5.0_openmpi_4.1.5

# ---------------------------------------------------------
# PIPELINE STAGE 1: WILDTYPE PROCESSING
# ---------------------------------------------------------
echo "Starting Wildtype Processing..."

# A. Center the protein in the box (1 = Protein, 0 = System)
echo "1 0" | gmx_mpi trjconv -s prod_1.tpr -f prod_1.xtc -o prod_center.xtc -pbc mol -center

# B. Remove rotation/translation (4 = Backbone, 0 = System)
echo "4 0" | gmx_mpi trjconv -s prod_1.tpr -f prod_center.xtc -o prod_final.xtc -fit rot+trans

# C. Calculate RMSD for the backbone (4 = Backbone for fit, 4 = Backbone for calc)
echo "4 4" | gmx_mpi rmsd -s prod_1.tpr -f prod_final.xtc -o rmsd_WT.xvg -tu ns

# D. Calculate per-residue RMSF for the backbone (4 = Backbone)
echo "4" | gmx_mpi rmsf -s prod_1.tpr -f prod_final.xtc -o rmsf_WT.xvg -res


# ---------------------------------------------------------
# PIPELINE STAGE 3: CLEANUP
# ---------------------------------------------------------
echo "Cleaning up intermediate centered files..."
# We delete the intermediate tumbling files to save your storage quota
rm -f prod_center.xtc

echo "Pipeline Complete! Download your .xvg files for plotting."
