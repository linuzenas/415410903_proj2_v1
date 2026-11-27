% GMMExample_2D.m
% Computer Vision (4105109), Project #2 - Gaussian Mixture Model with the EM algorithm
% Student: Petlu Linu Zenas Paul, ID 415410903
%
% The program was handed out with the project (the 1D companion file names Chris McCormick as
% its author). The code lines below are unchanged; the comments are my explanations.
%
% Model: every 2D point comes from one of k Gaussians. The density of the mixture is
%     p(x) = sum over j of  phi_j * N(x | mu_j, Sigma_j)
% where phi_j = prior probability of cluster j (they add up to 1), mu_j = its mean (1x2 row) and
% Sigma_j = its 2x2 covariance matrix. The program has four steps:
%   STEP 1: create data from two known Gaussians and draw it (the "truth")
%   STEP 2: choose starting values for phi, mu and Sigma
%   STEP 3: run EM: E-step (soft assignment of points) and M-step (new parameters), repeated
%   STEP 4: draw the estimated Gaussians over the data
% Needs the helper files gaussianND.m and weightedAverage.m in the same folder.

%%======================================================
%% STEP 1a: Generate data from two 2D distributions.

% True mean of Gaussian 1 as a row vector [x y]. Only used to create the data.
mu1 = [1 2];       % Mean
% True covariance of Gaussian 1 (2x2, symmetric): variance 3 in x, variance 2 in y,
% covariance 0.2 between x and y (the cloud is slightly tilted).
sigma1 = [ 3 .2;   % Covariance matrix
          .2  2];  % second row of sigma1: covariance .2 and variance 2 (closes the matrix)
% Number of points that will be drawn from Gaussian 1.
m1 = 200;          % Number of data points.

% Same three values for Gaussian 2: mean, covariance (x and y not correlated) and point count.
mu2 = [-1 -2];
sigma2 = [2 0;  % first row of sigma2: variance 2 in x, covariance 0
          0 1];  % second row of sigma2: covariance 0, variance 1 in y (closes the matrix)
m2 = 100;  % 100 points will come from Gaussian 2

% Generate sample points with the specified means and covariance matrices.
% chol gives the upper triangular matrix R1 with R1' * R1 = sigma1 (the "square root" of sigma1).
R1 = chol(sigma1);
% randn(m1, 2) is an m1 x 2 matrix of independent N(0,1) numbers (mean 0, covariance I).
% Multiplying by R1 changes the covariance to R1' * I * R1 = sigma1.
X1 = randn(m1, 2) * R1;
% repmat copies the row mu1 into an m1 x 2 matrix; adding it moves every point so that the
% mean of the cloud becomes mu1.
X1 = X1 + repmat(mu1, size(X1, 1), 1);

% Same procedure for Gaussian 2: Cholesky factor, correlated random numbers, shift by mu2.
R2 = chol(sigma2);
X2 = randn(m2, 2) * R2;  % correlated random numbers for Gaussian 2 (100 x 2)
X2 = X2 + repmat(mu2, size(X2, 1), 1);  % shift them so that their mean becomes mu2

% Put both point sets in one 300 x 2 matrix, one point per row. From now on the program does not
% know which point came from which Gaussian: that hidden label is what EM has to guess.
X = [X1; X2];

%%=====================================================
%% STEP 1b: Plot the data points and their pdfs.

% Open (or select) figure window number 1.
figure(1);

% Display a scatter plot of the two distributions.
% hold off: the next plot command clears the axes first.
hold off;
% Points of Gaussian 1 as blue circles (column 1 = x, column 2 = y).
plot(X1(:, 1), X1(:, 2), 'bo');
% hold on: later plots are added to the same axes instead of replacing the points.
hold on;
% Points of Gaussian 2 as red circles.
plot(X2(:, 1), X2(:, 2), 'ro');

% Make the figure background white.
set(gcf,'color','white') % White background for the figure.

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
% The pdf is evaluated on a regular 100 x 100 grid of points so that it can be drawn as contours.
gridSize = 100;
% 100 equally spaced values from -6 to 6; used for both the x and the y axis of the grid.
u = linspace(-6, 6, gridSize);
% meshgrid makes two 100 x 100 matrices: A holds the x value and B the y value of each grid point.
[A B] = meshgrid(u, u);
% A(:) and B(:) turn each matrix into one column of 10000 numbers; side by side they form the
% list of all 10000 grid points, one point per row.
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
% gaussianND returns a 10000 x 1 vector with the true pdf of Gaussian 1 at every grid point.
z1 = gaussianND(gridX, mu1, sigma1);
% The same for Gaussian 2.
z2 = gaussianND(gridX, mu2, sigma2);

% Reshape the responses back into a 2D grid to be plotted with contour.
% reshape fills column by column, the same order as A(:), so Z1(r,c) is the pdf at grid point (r,c).
Z1 = reshape(z1, gridSize, gridSize);
Z2 = reshape(z2, gridSize, gridSize);  % pdf of Gaussian 2 as a 100 x 100 grid

% Plot the contour lines to show the pdf over the data.
% contour draws lines of equal pdf height (the ellipses). C is the contour data and h the
% handle of the plot object; both are not used again.
[C, h] = contour(u, u, Z1);
[C, h] = contour(u, u, Z2);  % contour lines of true Gaussian 2

% Show x and y from -6 to 6 only, and give the figure a title.
axis([-6 6 -6 6])
title('Original Data and PDFs');  % title of figure 1

% This line is commented out in the original; it would write the height on the contour lines.
%set(h,'ShowText','on','TextStep',get(h,'LevelStep')*2);


%%====================================================
%% STEP 2: Choose initial values for the parameters.

% Set 'm' to the number of data points.
% size(X, 1) = number of rows of X = 300 points.
m = size(X, 1);

% k is the number of Gaussians in the model. It is chosen by me and not learned by EM.
k = 2;  % The number of clusters.
% n is the dimension of every point (2 coordinates).
n = 2;  % The vector lengths.

% Randomly select k data points to serve as the initial means.
% randperm(m) is a random ordering of the numbers 1..m, so every run starts differently.
indeces = randperm(m);
% Take the first k numbers of that ordering and use those k data points (rows of X) as the
% starting means. mu is k x 2: row j is the mean of cluster j.
mu = X(indeces(1:k), :);

% sigma will be a cell array: sigma{j} is the 2x2 covariance matrix of cluster j.
sigma = [];

% Use the overal covariance of the dataset as the initial variance for each cluster.
% Loop over the clusters j = 1..k.
for (j = 1 : k)
    % cov(X) is the covariance matrix of all 300 points; every cluster starts with the same wide
    % covariance that covers the whole data set.
    sigma{j} = cov(X);
% End of the loop over clusters.
end

% Assign equal prior probabilities to each cluster.
% phi is a 1 x k row, phi(j) = prior probability of cluster j. Start with 1/k each (0.5 and 0.5).
phi = ones(1, k) * (1 / k);

%%===================================================
%% STEP 3: Run Expectation Maximization

% Matrix to hold the probability that each data point belongs to each cluster.
% One row per data point, one column per cluster.
% W(i,j) = probability that point i belongs to cluster j (called "responsibility"). It is first
% filled with zeros (300 x 2); every row will sum to 1 once it is computed.
W = zeros(m, k);

% Loop until convergence.
% Repeat E-step and M-step at most 1000 times; the loop is left earlier by break (see the end).
for (iter = 1:1000)
    
    % Print the number of the current iteration.
    fprintf('  EM Iteration %d\n', iter);

    %%===============================================
    %% STEP 3a: Expectation
    %
    % Calculate the probability for each data point for each distribution.
    
    % Matrix to hold the pdf value for each every data point for every cluster.
    % One row per data point, one column per cluster.
    % pdf(i,j) will be N(x_i | mu_j, Sigma_j), the height of Gaussian j at point i.
    pdf = zeros(m, k);
    
    % For each cluster...
    for (j = 1 : k)
        
        % Evaluate the Gaussian for all data points for cluster 'j'.
        % gaussianND returns all 300 values at once (a 300 x 1 vector) using the current mean of
        % cluster j (row j of mu) and its current covariance sigma{j}; it fills column j of pdf.
        pdf(:, j) = gaussianND(X, mu(j, :), sigma{j});
    % End of the loop over clusters.
    end
    
    % Multiply each pdf value by the prior probability for cluster.
    %    pdf  [m  x  k]
    %    phi  [1  x  k]   
    %  pdf_w  [m  x  k]
    % bsxfun(@times, ...) multiplies column j of pdf by phi(j) (phi is stretched over all rows).
    % pdf_w(i,j) = phi_j * N(x_i | mu_j, Sigma_j): how likely point i is AND from cluster j.
    pdf_w = bsxfun(@times, pdf, phi);
    
    % Divide the weighted probabilities by the sum of weighted probabilities for each cluster.
    %   sum(pdf_w, 2) -- sum over the clusters.
    % sum(pdf_w, 2) adds the k numbers of every row: p(x_i), the density of point i under the
    % whole mixture (a 300 x 1 vector). Dividing every row by its own sum gives Bayes' rule:
    %    W(i,j) = phi_j N(x_i|mu_j,Sigma_j) / sum over l of phi_l N(x_i|mu_l,Sigma_l)
    % so every row of W sums to 1.
    W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));
    
    %%===============================================
    %% STEP 3b: Maximization
    %%
    %% Calculate the probability for each data point for each distribution.

    % Store the previous means so we can check for convergence.
    % Remember the means used in this iteration so they can be compared with the new ones below.
    prevMu = mu;    
    
    % For each of the clusters...
    for (j = 1 : k)
    
        % Calculate the prior probability for cluster 'j'.
        % New prior = average responsibility of cluster j = (1/m) * sum over i of W(i,j), the
        % share of all points that cluster j is responsible for.
        phi(j) = mean(W(:, j), 1);
        
        % Calculate the new mean for cluster 'j' by taking the weighted
        % average of *all* data points.
        % mu_j = sum_i W(i,j) x_i / sum_i W(i,j): every point counts with its responsibility.
        % weightedAverage returns a 1 x 2 row, which is stored in row j of mu.
        mu(j, :) = weightedAverage(W(:, j), X);

        % Calculate the covariance matrix for cluster 'j' by taking the 
        % weighted average of the covariance for each training example. 
        
        % sigma_k collects the sum of the weighted covariance terms; it starts as a 2 x 2 zero matrix.
        sigma_k = zeros(n, n);
        
        % Subtract the cluster mean from all data points.
        % Xm(i,:) = x_i - mu_j (using the NEW mean just computed); Xm is 300 x 2.
        Xm = bsxfun(@minus, X, mu(j, :));
        
        % Calculate the contribution of each training example to the covariance matrix.
        % Loop over all data points i.
        for (i = 1 : m)
            % Xm(i,:)' is a 2 x 1 column and Xm(i,:) a 1 x 2 row, so their product is the 2 x 2
            % matrix (x_i - mu_j)(x_i - mu_j)'. It is multiplied by the weight W(i,j) and added up.
            sigma_k = sigma_k + (W(i, j) .* (Xm(i, :)' * Xm(i, :)));
        % End of the loop over data points.
        end
        
        % Divide by the sum of weights.
        % Sigma_j = sum_i W(i,j) (x_i-mu_j)(x_i-mu_j)' / sum_i W(i,j): a weighted covariance.
        sigma{j} = sigma_k ./ sum(W(:, j));
    % End of the loop over clusters (M-step finished for all clusters).
    end
    
    % Check for convergence.
    % mu == prevMu compares the two k x 2 matrices element by element. The condition is true only
    % if no number changed at all. Only the means are tested, not phi and sigma.
    if (mu == prevMu)
        % The means did not change: leave the EM loop.
        break
    % End of the if statement.
    end

% End of Expectation Maximization loop.    
end

%%=====================================================
%% STEP 4: Plot the data points and their estimated pdfs.

% Display a scatter plot of the two distributions.
% Open figure window number 2 (a new window, so figure 1 stays unchanged).
figure(2);
% Clear the axes before the next plot.
hold off;
% Draw the data again: points of Gaussian 1 in blue, points of Gaussian 2 in red.
plot(X1(:, 1), X1(:, 2), 'bo');
hold on;  % keep the blue points and add to the same axes
plot(X2(:, 1), X2(:, 2), 'ro');  % points of Gaussian 2 as red circles

% White figure background.
set(gcf,'color','white') % White background for the figure.

% Mark the TRUE means mu1 and mu2 (not the estimated ones) with black crosses.
plot(mu1(1), mu1(2), 'kx');
plot(mu2(1), mu2(2), 'kx');  % true mean of Gaussian 2 as a black cross

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
% Same grid as in STEP 1b: 100 x 100 points between -6 and 6, stored as 10000 rows [x y].
gridSize = 100;
u = linspace(-6, 6, gridSize);  % 100 values from -6 to 6 for the x and y axis
[A B] = meshgrid(u, u);  % x (A) and y (B) coordinates of all grid points
gridX = [A(:), B(:)];  % list of the 10000 grid points, one per row

% Calculate the Gaussian response for every value in the grid.
% This time the ESTIMATED means mu(j,:) and covariances sigma{j} from EM are used. Note that the
% pdf of each component is drawn alone, without multiplying it by phi_j.
z1 = gaussianND(gridX, mu(1, :), sigma{1});
z2 = gaussianND(gridX, mu(2, :), sigma{2});  % estimated pdf of cluster 2 on the grid

% Reshape the responses back into a 2D grid to be plotted with contour.
Z1 = reshape(z1, gridSize, gridSize);
Z2 = reshape(z2, gridSize, gridSize);  % reshape estimated pdf of cluster 2 to 100 x 100

% Plot the contour lines to show the pdf over the data.
% Draw the contour lines (ellipses) of the two estimated Gaussians over the data.
[C, h] = contour(u, u, Z1);
[C, h] = contour(u, u, Z2);  % contour lines of estimated cluster 2
% Limit the axes to -6..6 and add the title.
axis([-6 6 -6 6])

title('Original Data and Estimated PDFs');  % title of figure 2
