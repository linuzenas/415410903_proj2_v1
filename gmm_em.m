function res = gmm_em(X, k, opt)
% EM for a Gaussian mixture. The loop is the same as STEP 3 of GMMExample_2D.m,
% but the starting values and the stopping rule can be changed through 'opt':
%   opt.mu0     k x n matrix with the initial means        (default: k random data points)
%   opt.sigma0  1 x k cell with the initial covariances    (default: cov(X) for every cluster)
%   opt.phi0    1 x k initial mixing weights               (default: 1/k each)
%   opt.maxIter maximum number of iterations               (default: 1000)
%   opt.tol     stop when the largest change of mu < tol   (default 0 = exact equality, as in the original)
% It also records the log-likelihood of the data at every iteration.
    [m, n] = size(X);
    if ~isfield(opt, 'maxIter'), opt.maxIter = 1000; end
    if ~isfield(opt, 'tol'),     opt.tol = 0;        end

    indeces = randperm(m);                       % always drawn, so the random stream matches the original script
    if isfield(opt, 'mu0'),    mu = opt.mu0;    else mu = X(indeces(1:k), :); end
    sigma = cell(1, k);
    for j = 1:k
        if isfield(opt, 'sigma0'), sigma{j} = opt.sigma0{j}; else sigma{j} = cov(X); end
    end
    if isfield(opt, 'phi0'),   phi = opt.phi0;  else phi = ones(1, k) * (1 / k); end
    res.mu_init = mu; res.phi_init = phi; res.sigma_init = sigma;

    ll = zeros(1, 0);
    for iter = 1:opt.maxIter
        % ---- E-step
        pdf = zeros(m, k);
        for j = 1:k
            pdf(:, j) = gaussianND(X, mu(j, :), sigma{j});
        end
        pdf_w = bsxfun(@times, pdf, phi);
        ll(end + 1) = sum(log(sum(pdf_w, 2)));   % log-likelihood with the parameters used in this E-step
        W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));

        % ---- M-step
        prevMu = mu;
        for j = 1:k
            phi(j) = mean(W(:, j), 1);
            mu(j, :) = weightedAverage(W(:, j), X);
            Xm = bsxfun(@minus, X, mu(j, :));
            sigma_k = zeros(n, n);
            for i = 1:m
                sigma_k = sigma_k + (W(i, j) .* (Xm(i, :)' * Xm(i, :)));
            end
            sigma{j} = sigma_k ./ sum(W(:, j));
        end

        % ---- stopping rule
        if opt.tol == 0
            done = all(mu(:) == prevMu(:));
        else
            done = max(abs(mu(:) - prevMu(:))) < opt.tol;
        end
        if done, break; end
    end
    res.phi = phi; res.mu = mu; res.sigma = sigma; res.iter = iter; res.ll = ll; res.W = W;
end
