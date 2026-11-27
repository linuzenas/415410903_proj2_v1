%% GMM_experiments.m  - changing the parameters of the GMM / EM program (project part 2)
% Needs: gaussianND.m, weightedAverage.m, gmm_em.m, gen_data.m, plot_gmm.m, set_seed.m, save_fig.m
% Everything is seeded, so the numbers are the same every time the script is run.
clear; close all;
isOct = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if ~isempty(getenv('GMM_HEADLESS')) && isOct          % only used when there is no screen
    graphics_toolkit('gnuplot'); set(0, 'defaultfigurevisible', 'off');
end
outdir = fullfile('results', 'experiment_figures'); if ~exist('results', 'dir'), mkdir('results'); end
if ~exist(outdir, 'dir'), mkdir(outdir); end

% true parameters used in the project (same as GMMExample_2D.m)
mu1 = [1 2];  s1 = [3 .2; .2 2];  m1 = 200;
mu2 = [-1 -2]; s2 = [2 0; 0 1];   m2 = 100;

%% ---------------------------------------------------------------- E0 baseline
fprintf('\n===== E0  baseline (same seed as the original run) =====\n');
set_seed(1); [X, lab] = gen_data(mu1, s1, m1, mu2, s2, m2);
r0 = gmm_em(X, 2, struct());
fprintf('iterations: %d, final log-likelihood: %.4f\n', r0.iter, r0.ll(end));
fprintf('estimated phi: %.4f %.4f\n', r0.phi);
fprintf('estimated mu1: %.4f %.4f   mu2: %.4f %.4f\n', r0.mu(1,:), r0.mu(2,:));
fprintf('estimated Sigma1: [%.4f %.4f; %.4f %.4f]\n', r0.sigma{1}(:));
fprintf('estimated Sigma2: [%.4f %.4f; %.4f %.4f]\n', r0.sigma{2}(:));
Xa = X(lab==1,:); Xb = X(lab==2,:);
fprintf('--- for comparison, statistics of the points that really came from each Gaussian ---\n');
fprintf('true phi: %.4f %.4f   (fraction of points: %.4f %.4f)\n', 2/3, 1/3, mean(lab==1), mean(lab==2));
fprintf('sample mean 1: %.4f %.4f   sample mean 2: %.4f %.4f\n', mean(Xa), mean(Xb));
fprintf('sample cov 1: [%.4f %.4f; %.4f %.4f]\n', cov(Xa, 1)(:));
fprintf('sample cov 2: [%.4f %.4f; %.4f %.4f]\n', cov(Xb, 1)(:));
fprintf('log-likelihood of the data if the true labels were used: %.4f\n', ...
    sum(log(mean(lab==1)*gaussianND(X,mean(Xa),cov(Xa,1)) + mean(lab==2)*gaussianND(X,mean(Xb),cov(Xb,1)))));
fprintf('number of points whose responsibility W(i,1) > 0.5 but that came from Gaussian 2: %d\n', sum(r0.W(:,1) > 0.5 & lab == 2));
fprintf('number of points whose responsibility W(i,2) > 0.5 but that came from Gaussian 1: %d\n', sum(r0.W(:,2) > 0.5 & lab == 1));

figure; plot(1:r0.iter, r0.ll, 'k-'); grid on;
xlabel('EM iteration'); ylabel('log-likelihood of the data'); title('E0: log-likelihood during EM');
save_fig(fullfile(outdir, 'fig_E0_loglik.png'), 560, 380);
fprintf('log-likelihood never decreased: %d (largest decrease between two iterations: %.3g)\n', all(diff(r0.ll) >= -1e-9), min([0 diff(r0.ll)]));
for it = [1 2 5 10 20 50 100 200]
    fprintf('  after %3d iterations log-likelihood = %.4f\n', it, r0.ll(it));
end

%% ---------------------------------------------------------------- E4 stopping early
fprintf('\n===== E4  what EM looks like after a few iterations (same start as E0) =====\n');
figure; set(gcf, 'color', 'white'); stops = [1 5 20 r0.iter];
for q = 1:4
    set_seed(1); gen_data(mu1, s1, m1, mu2, s2, m2);
    rr = gmm_em(X, 2, struct('maxIter', stops(q)));
    subplot(2, 2, q); plot_gmm(X, lab, rr.mu, rr.sigma, sprintf('after %d iterations', rr.iter));
    fprintf('after %3d iterations: phi = %.4f %.4f  mu1 = (%.3f, %.3f)  mu2 = (%.3f, %.3f)\n', rr.iter, rr.phi, rr.mu(1,:), rr.mu(2,:));
end
save_fig(fullfile(outdir, 'fig_E4_progress.png'), 900, 700);
fprintf('initial means were: (%.3f, %.3f) and (%.3f, %.3f)\n', r0.mu_init(1,:), r0.mu_init(2,:));

%% ---------------------------------------------------------------- E1 different random starts
fprintf('\n===== E1  same data, 20 different random initialisations (component A = the one near (1,2)) =====\n');
fprintf('seed | iters |  final LL  |  phiA  |  muA              |  muB\n');
LLs = zeros(1, 20); its = zeros(1, 20);
for s = 1:20
    set_seed(1); gen_data(mu1, s1, m1, mu2, s2, m2);
    set_seed(100 + s); rr = gmm_em(X, 2, struct());
    [~, a] = min(sum((rr.mu - repmat(mu1, 2, 1)).^2, 2)); b = 3 - a;
    LLs(s) = rr.ll(end); its(s) = rr.iter;
    fprintf('%4d | %5d | %10.4f | %.4f | (%.4f, %.4f) | (%.4f, %.4f)\n', 100 + s, rr.iter, rr.ll(end), rr.phi(a), rr.mu(a,:), rr.mu(b,:));
end
fprintf('distinct final log-likelihoods (rounded to 0.001): %d\n', numel(unique(round(LLs * 1000))));
fprintf('iterations: min %d, max %d, mean %.1f\n', min(its), max(its), mean(its));

%% ---------------------------------------------------------------- E1b different data sets
fprintf('\n===== E1b  different random data sets (true values: phi = 0.667/0.333, mu = (1,2) / (-1,-2)) =====\n');
fprintf('seed | iters | phiA  | muA               | muB\n');
for s = 11:15
    set_seed(s); [Xs, ls] = gen_data(mu1, s1, m1, mu2, s2, m2);
    rr = gmm_em(Xs, 2, struct());
    [~, a] = min(sum((rr.mu - repmat(mu1, 2, 1)).^2, 2)); b = 3 - a;
    fprintf('%4d | %5d | %.4f | (%.4f, %.4f) | (%.4f, %.4f)\n', s, rr.iter, rr.phi(a), rr.mu(a,:), rr.mu(b,:));
end

%% ---------------------------------------------------------------- E2 number of clusters
fprintf('\n===== E2  number of clusters k (same data and seed; stop when mu changes < 1e-8) =====\n');
fprintf(' k | iters |  final LL  | free parameters |   BIC\n');
figure; set(gcf, 'color', 'white'); kk = 1:5; panel = 0;
for k = kk
    set_seed(1); gen_data(mu1, s1, m1, mu2, s2, m2);
    set_seed(100); rr = gmm_em(X, k, struct('tol', 1e-8));
    p = 6 * k - 1; bic = -2 * rr.ll(end) + p * log(size(X, 1));
    fprintf('%2d | %5d | %10.4f | %15d | %9.2f\n', k, rr.iter, rr.ll(end), p, bic);
    if any(k == [1 3 4 5])
        panel = panel + 1; subplot(2, 2, panel);
        plot_gmm(X, lab, rr.mu, rr.sigma, sprintf('k = %d', k));
    end
    if k == 3, fprintf('   k=3: phi = %s\n', mat2str(rr.phi, 3)); end
end
save_fig(fullfile(outdir, 'fig_E2_k.png'), 900, 700);

%% ---------------------------------------------------------------- E3 initial values
fprintf('\n===== E3  different initial values (k = 2, same data) =====\n');
cases = {};
cases{end+1} = {'both means start at the same point (0,0)', struct('mu0', [0 0; 0 0]), 'both means start at (0,0)'};
cases{end+1} = {'means start at (0,0) and (0.01,0.01)',     struct('mu0', [0 0; 0.01 0.01]), 'means start 0.01 apart'};
cases{end+1} = {'means start at the true means',            struct('mu0', [1 2; -1 -2]), 'means start at true means'};
cases{end+1} = {'both means start inside the upper cluster', struct('mu0', [1 2; 1.5 2.5]), 'both means in upper cluster'};
cases{end+1} = {'means start swapped: (-1,-2) and (1,2)',   struct('mu0', [-1 -2; 1 2]), 'means start swapped'};
cases{end+1} = {'initial covariance 0.01*I (too small)',    struct('sigma0', {{0.01*eye(2), 0.01*eye(2)}}), 'initial Sigma = 0.01 I'};
cases{end+1} = {'initial covariance 100*I (very large)',    struct('sigma0', {{100*eye(2), 100*eye(2)}}), 'initial Sigma = 100 I'};
cases{end+1} = {'initial phi = (0.95, 0.05)',               struct('phi0', [0.95 0.05]), 'initial phi = (0.95, 0.05)'};
figure; set(gcf, 'color', 'white'); panel = 0;
for c = 1:numel(cases)
    set_seed(1); gen_data(mu1, s1, m1, mu2, s2, m2);
    set_seed(100); ws = warning('off', 'all'); rr = gmm_em(X, 2, cases{c}{2}); warning(ws);
    bad = any(isnan(rr.phi)) || any(isnan(rr.mu(:)));
    fprintf('\n%s\n', cases{c}{1});
    if bad
        fprintf('  RESULT: NaN after %d iterations (phi = %s)\n', rr.iter, mat2str(rr.phi, 3));
    else
        fprintf('  iterations %d, final LL %.4f, phi = %.4f %.4f, mu1 = (%.3f, %.3f), mu2 = (%.3f, %.3f)\n', rr.iter, rr.ll(end), rr.phi, rr.mu(1,:), rr.mu(2,:));
        fprintf('  Sigma1 = [%.3f %.3f; %.3f %.3f]   Sigma2 = [%.3f %.3f; %.3f %.3f]\n', rr.sigma{1}(:), rr.sigma{2}(:));
        if any(c == [1 4 7 8])
            panel = panel + 1; subplot(2, 2, panel); plot_gmm(X, lab, rr.mu, rr.sigma, cases{c}{3});
        end
    end
end
save_fig(fullfile(outdir, 'fig_E3_init.png'), 900, 700);

%% ---------------------------------------------------------------- E5 stopping rule
fprintf('\n===== E5  stopping rule (same data and start as E0) =====\n');
fprintf('rule                      | iters | final LL    | muA\n');
tols = [1e-2 1e-4 1e-6 1e-10 0];
for q = 1:numel(tols)
    set_seed(1); gen_data(mu1, s1, m1, mu2, s2, m2);
    rr = gmm_em(X, 2, struct('tol', tols(q)));
    if tols(q) == 0, nm = 'exact equality (original)'; else nm = sprintf('max change of mu < %g', tols(q)); end
    fprintf('%-25s | %5d | %11.6f | (%.5f, %.5f)\n', nm, rr.iter, rr.ll(end), rr.mu(1,:));
end

%% ---------------------------------------------------------------- E6 changing the data
fprintf('\n===== E6  changing the data that is generated =====\n');
data = {};
data{end+1} = {'second Gaussian moved to mean (0,0)  (clusters overlap more)', [1 2], s1, 200, [0 0], s2, 100, 'second mean (0,0)'};
data{end+1} = {'second Gaussian moved to mean (0.5,0.5) (strong overlap)',     [1 2], s1, 200, [0.5 0.5], s2, 100, 'second mean (0.5,0.5)'};
data{end+1} = {'unbalanced sizes: 280 and 20 points',                           [1 2], s1, 280, [-1 -2], s2, 20, '280 and 20 points'};
data{end+1} = {'more data: 1000 and 500 points',                                [1 2], s1, 1000, [-1 -2], s2, 500, '1000 and 500 points'};
figure; set(gcf, 'color', 'white');
for d = 1:numel(data)
    D = data{d};
    set_seed(1); [Xd, ld] = gen_data(D{2}, D{3}, D{4}, D{5}, D{6}, D{7});
    set_seed(100); rr = gmm_em(Xd, 2, struct('tol', 1e-8));
    [~, a] = min(sum((rr.mu - repmat(D{2}, 2, 1)).^2, 2)); b = 3 - a;
    fprintf('\n%s\n', D{1});
    fprintf('  true  phi = %.3f %.3f, mu = (%.2f, %.2f) and (%.2f, %.2f)\n', D{4}/(D{4}+D{7}), D{7}/(D{4}+D{7}), D{2}, D{5});
    fprintf('  found phi = %.3f %.3f, mu = (%.3f, %.3f) and (%.3f, %.3f), iterations %d\n', rr.phi(a), rr.phi(b), rr.mu(a,:), rr.mu(b,:), rr.iter);
    fprintf('  Sigma A = [%.3f %.3f; %.3f %.3f]   Sigma B = [%.3f %.3f; %.3f %.3f]\n', rr.sigma{a}(:), rr.sigma{b}(:));
    subplot(2, 2, d); plot_gmm(Xd, ld, rr.mu, rr.sigma, D{8});
end
save_fig(fullfile(outdir, 'fig_E6_data.png'), 900, 700);
fprintf('\nall experiments finished\n');
