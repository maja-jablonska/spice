#!/bin/bash

# Define parameter grids
inclinations=(85.0 90.0)
periods=(3.0 10.0 30.0)
qs=(0.5 1.0)
eccs=(0.0 0.3 0.6)
primary_masses=(1.0)
n_times=100
<<<<<<< HEAD
<<<<<<< HEAD
output_path="scr_mk27/eclipse_grid_results"
=======
output_path="eclipse_grid_results"
>>>>>>> 5115cc6c5a2747447e8dd26acc4501851ea5c7da
=======
output_path="eclipse_grid_results"
>>>>>>> 5115cc6c5a2747447e8dd26acc4501851ea5c7da

# Create output directory
mkdir -p "$output_path"

# Path to the script
script_path="tutorial/paper_results/check_phoebe_eclipses.py"

# Loop over all parameter combinations
for incl in "${inclinations[@]}"; do
  for period in "${periods[@]}"; do
    for q in "${qs[@]}"; do
      for ecc in "${eccs[@]}"; do
        for mass in "${primary_masses[@]}"; do
          echo "----------------------------------------------------------------"
          echo "Running with: inclination=$incl, period=$period, q=$q, ecc=$ecc, primary_mass=$mass"
          echo "----------------------------------------------------------------"
          
          python "$script_path" \
            "$incl" \
            "$period" \
            --n_times "$n_times" \
            --q "$q" \
            --ecc "$ecc" \
            --primary_mass "$mass" \
            --output_path "$output_path"
            
          # Check exit code
          if [ $? -ne 0 ]; then
            echo "Error running script with parameters: inclination=$incl, period=$period, q=$q, ecc=$ecc, primary_mass=$mass"
            echo "Aborting."
            exit 1
          fi
        done
      done
    done
  

  done
done

echo "All grid runs completed successfully."
