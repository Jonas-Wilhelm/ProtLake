#!/usr/bin/bash

protlake_dir="tests_out/af3/af3_analysis/testdata_af3_protlake"
mkdir -p "$(dirname "$protlake_dir")"

# copy the test protlake to a differnt directory before modifying it
rm -rf "$protlake_dir"
cp -r tests/af3/testdata_af3_protlake "$protlake_dir"

python -m  analyze.af3.analyze \
    --num-tasks 2 \
    --protlake-path "$protlake_dir" \
    --design-dir tests/af3/testdata_design_models \
    --atom_names_rmsd 'MY-AKN:TS1$C1:C2:C3:O1:C4:C5:C6:C7:C8:C9:C10 MY-AZD:TS1$N3:N4:N5:C16:C17:C18:C19:C20:C21:N6:N7' \
    --chem_eq_atoms 'MY-AKN:TS1$C5:C6_C7:C8' \
    --custom-scorefxn-dir analyze/af3/scorefxns \
    --ligand_atom_distances MY-AKN:C1_MY-AZD:N3 MY-AKN:C2_MY-AZD:N5 \
    --log-dir tests_out/logs

