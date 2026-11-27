Computer Vision (4105109) - Project #2, Gaussian Mixture Model (EM algorithm)
Student ID: 415410903     Name: Petlu Linu Zenas Paul

Files
-----
GMMExample_2D.m            the project script with my comments on every line (code lines unchanged)
gaussianND.m               helper: Gaussian density for many points (commented)
weightedAverage.m          helper: weighted mean (commented)
run_baseline.m             runs GMMExample_2D.m with a fixed random seed and saves the two figures
trace_first_iteration.m    prints the numbers of the first EM iteration (used in the report, part 1)
GMM_experiments.m          experiments E0-E6 with changed parameters (used in the report, part 2)
check_local_optimum.m      two small checks mentioned in the report (section 2.5)
gmm_em.m, gen_data.m,      small functions used by the experiment scripts (same EM code as the
plot_gmm.m, set_seed.m,    project script, but with settings that can be changed)
save_fig.m
results/                   output of the scripts: figures, numbers and logs
  fig1_original_data_and_pdfs.png, fig2_estimated_pdfs.png, baseline_final_parameters.txt
  trace_first_iteration_output.txt, check_local_optimum_output.txt, experiments_log.txt
  experiment_figures/      figures of the experiments
The report is the .docx file.

How to run
----------
The code was written for MATLAB, and I ran everything with GNU Octave 8.4.0 (free, octave.org).
No extra packages are needed. In the Octave or MATLAB command window:

    cd 'C:\path\to\this\folder'
    run_baseline            % project script with seed 1; writes the two figures into results/
    trace_first_iteration
    check_local_optimum
    GMM_experiments         % takes about 2 minutes; writes the log numbers and figures

To run the project script alone, type GMMExample_2D. It uses random numbers (randn, randperm),
so without the seed every run gives slightly different figures and numbers.

The random number generators of Octave and MATLAB are different. The numbers in the report
(for example 337 iterations for seed 1) come from Octave 8.4.0; MATLAB will give similar but not
identical numbers.
