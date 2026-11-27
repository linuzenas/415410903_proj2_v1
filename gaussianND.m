function [ pdf ] = gaussianND(X, mu, Sigma)
% gaussianND  - value of the n-dimensional Gaussian density N(x | mu, Sigma) at many points.
%      X - m x n matrix of points, one point per row.
%     mu - 1 x n row vector with the mean.
%  Sigma - n x n covariance matrix.
% Returns pdf, an m x 1 column: pdf(i) is the density at point X(i,:),
%    N(x|mu,Sigma) = (2*pi)^(-n/2) * det(Sigma)^(-1/2) * exp( -1/2 (x-mu)' inv(Sigma) (x-mu) )

% n = number of columns of X = dimension of the points (2 in this project).
% Get the vector length.
n = size(X, 2);

% Subtract the mean from every data point.
% meanDiff(i,:) = X(i,:) - mu, an m x n matrix. bsxfun stretches the row mu over all m rows.
meanDiff = bsxfun(@minus, X, mu);

% Calculate the multivariate gaussian.
% The factor 1/sqrt((2*pi)^n * det(Sigma)) is the normalising constant that makes the total
% volume under the density equal to 1. In the exponent, meanDiff * inv(Sigma) is m x n; the
% element-wise product with meanDiff and the sum over each row (sum(..., 2)) give the number
% (x_i-mu) * inv(Sigma) * (x_i-mu)' for every row i. Multiplying by -1/2 and applying exp gives the
% bell shape. (* and .* have the same priority and are evaluated from left to right.)
pdf = 1 / sqrt((2*pi)^n * det(Sigma)) * exp(-1/2 * sum((meanDiff * inv(Sigma) .* meanDiff), 2));

end
