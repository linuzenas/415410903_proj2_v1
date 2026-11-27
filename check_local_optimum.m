%% check_local_optimum.m - two small checks used in the report (section 2.5)
% (a) Is the "poor" solution of experiment E3 (initial phi = 0.95/0.05) really a local optimum, or
%     was the 1000-iteration limit just too short?   (b) Why does the start with Sigma = 0.01*I give NaN?
mu1 = [1 2]; s1 = [3 .2; .2 2]; m1 = 200; mu2 = [-1 -2]; s2 = [2 0; 0 1]; m2 = 100;
set_seed(1); [X, lab] = gen_data(mu1, s1, m1, mu2, s2, m2);

fprintf('(a) start with phi0 = (0.95, 0.05), stop when mu changes by less than 1e-10 (at most 6000 iterations)\n');
set_seed(100); r = gmm_em(X, 2, struct('phi0', [0.95 0.05], 'maxIter', 6000, 'tol', 1e-10));
for it = [100 500 1000 length(r.ll)]
    fprintf('    iteration %5d: log-likelihood %.6f\n', it, r.ll(it));
end
fprintf('    stopped after %d iterations, phi = %.4f %.4f\n', r.iter, r.phi);

fprintf('(b) first E-step with Sigma = 0.01*I\n');
set_seed(100);
m = size(X, 1); indeces = randperm(m); mu = X(indeces(1:2), :);
pdf = zeros(m, 2);
ws = warning('off', 'all');
for j = 1:2, pdf(:, j) = gaussianND(X, mu(j, :), 0.01 * eye(2)); end
warning(ws);
pdf_w = bsxfun(@times, pdf, [0.5 0.5]);
fprintf('    points whose weighted density is exactly 0 for BOTH clusters: %d of %d\n', sum(sum(pdf_w, 2) == 0), m);
fprintf('    -> W = 0/0 = NaN for those points, and the NaN spreads to mu, sigma and phi in the M-step\n');
