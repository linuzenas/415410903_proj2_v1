%% trace_first_iteration.m - prints the numbers of the first EM iteration (same data and start as the baseline run)
% Needs gaussianND.m, weightedAverage.m, gen_data.m, set_seed.m
mu1 = [1 2]; s1 = [3 .2; .2 2]; m1 = 200; mu2 = [-1 -2]; s2 = [2 0; 0 1]; m2 = 100;
set_seed(1); [X, lab] = gen_data(mu1, s1, m1, mu2, s2, m2);
m = size(X, 1); k = 2; n = 2;
indeces = randperm(m); mu = X(indeces(1:k), :);
sigma = cell(1, k); for j = 1:k, sigma{j} = cov(X); end
phi = ones(1, k) / k;
fprintf('data size: %d x %d, first point x1 = (%.4f, %.4f)\n', size(X), X(1,:));
fprintf('initial mu:\n'); disp(mu); fprintf('initial Sigma (= cov(X), same for both clusters):\n'); disp(sigma{1}); fprintf('initial phi: %.2f %.2f\n', phi);
pdf = zeros(m, k); for j = 1:k, pdf(:, j) = gaussianND(X, mu(j, :), sigma{j}); end
pdf_w = bsxfun(@times, pdf, phi); W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));
fprintf('\nE-step for the first data point:\n');
fprintf('  pdf(1,:)   = %.6f  %.6f\n', pdf(1,:));
fprintf('  pdf_w(1,:) = %.6f  %.6f\n', pdf_w(1,:));
fprintf('  sum        = %.6f\n', sum(pdf_w(1,:)));
fprintf('  W(1,:)     = %.4f  %.4f  (sums to %.4f)\n', W(1,:), sum(W(1,:)));
fprintf('column sums of W (soft number of points per cluster): %.3f %.3f (total %d)\n', sum(W, 1), m);
fprintf('log-likelihood before the update: %.4f\n', sum(log(sum(pdf_w, 2))));
fprintf('\nM-step:\n');
for j = 1:k
    phiN = mean(W(:, j), 1); muN = weightedAverage(W(:, j), X);
    Xm = bsxfun(@minus, X, muN); S = zeros(n, n);
    for i = 1:m, S = S + W(i, j) .* (Xm(i, :)' * Xm(i, :)); end
    S = S ./ sum(W(:, j));
    fprintf('  cluster %d: phi = %.4f, mu = (%.4f, %.4f), Sigma = [%.4f %.4f; %.4f %.4f]\n', j, phiN, muN, S(:));
end
