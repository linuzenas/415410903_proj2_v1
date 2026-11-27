function [X, labels] = gen_data(mu1, sigma1, m1, mu2, sigma2, m2)
% Same data generation as STEP 1a of GMMExample_2D.m, packed into a function.
% labels(i) = 1 or 2 tells which Gaussian produced point i (used only for plotting/checking).
    R1 = chol(sigma1);
    X1 = randn(m1, 2) * R1;
    X1 = X1 + repmat(mu1, size(X1, 1), 1);

    R2 = chol(sigma2);
    X2 = randn(m2, 2) * R2;
    X2 = X2 + repmat(mu2, size(X2, 1), 1);

    X = [X1; X2];
    labels = [ones(m1, 1); 2 * ones(m2, 1)];
end
