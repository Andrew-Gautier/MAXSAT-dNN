#!/bin/bash
# filename: run_benchmarks.sh

# Output CSV file
OUTPUT_FILE="amai-maxsat-results(5s).csv'"
TIME_LIMIT=5.0
THREADS=4
SOLVERS=("CP-SAT-MAXSAT" "Gurobi-MAXSAT" "MaxSatdNN")
REPETITIONS=5
ISOLATE_FLAG="--isolate"

# Directory containing CNF files
CNF_DIR="instances/vlsat2"
# Clear the output file if it exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Creating new results file..."
else
    echo "Appending to existing results file..."
fi

# Get the first CNF file for warmup
first_cnf=$(ls $CNF_DIR/*.cnf | head -n 1)
if [ -n "$first_cnf" ]; then
    echo "Running warmup passes on $first_cnf..."
    for i in {1..3}; do
        echo "Warmup run $i/3"
    python3 Benchmark.py --time_limit $TIME_LIMIT --threads $THREADS --output /dev/null "$first_cnf" --solvers "${SOLVERS[@]}" $ISOLATE_FLAG --repetitions 1
    done
    echo "Warmup complete."
fi

# Process each CNF file
for cnf_file in $CNF_DIR/*.cnf; do
  echo "Processing $cnf_file..."
    python3 Benchmark.py --time_limit $TIME_LIMIT --threads $THREADS --output $OUTPUT_FILE "$cnf_file" --solvers "${SOLVERS[@]}" $ISOLATE_FLAG --repetitions $REPETITIONS
  echo "Done with $cnf_file"
  echo "-----------------------------------"
done

echo "All benchmarks completed. Results saved to $OUTPUT_FILE"