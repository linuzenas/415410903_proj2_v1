%% run_baseline.m - runs GMMExample_2D.m with a fixed random seed and saves the two figures.
% The original script uses random numbers (randn, randperm), so every plain run looks a little
% different. With the seed below the run can be repeated exactly (GNU Octave 8.4.0, seed 1).
isOct = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if ~isempty(getenv('GMM_HEADLESS')) && isOct      % only needed when there is no screen
    graphics_toolkit('gnuplot'); setenv('GNUTERM', 'png'); set(0, 'defaultfigurevisible', 'off');
end
if ~exist('results', 'dir'), mkdir('results'); end
set_seed(1);
GMMExample_2D;                                    % the commented project script
figure(1); save_fig(fullfile('results', 'fig1_original_data_and_pdfs.png'), 640, 480);
figure(2); save_fig(fullfile('results', 'fig2_estimated_pdfs.png'), 640, 480);
fprintf('\nFINAL phi:\n');        disp(phi);
fprintf('FINAL mu (row j = cluster j):\n'); disp(mu);
fprintf('FINAL sigma{1}:\n');    disp(sigma{1});
fprintf('FINAL sigma{2}:\n');    disp(sigma{2});
fprintf('iterations used: %d\n', iter);
