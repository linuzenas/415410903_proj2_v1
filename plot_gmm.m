function plot_gmm(X, labels, mu, sigma, titleStr)
% Scatter plot of the data (coloured by the TRUE cluster) plus contour lines of every
% estimated component and a black x at every estimated mean. Same style as figure 2 of the project.
    hold off;
    plot(X(labels == 1, 1), X(labels == 1, 2), 'bo'); hold on;
    plot(X(labels == 2, 1), X(labels == 2, 2), 'ro');
    gridSize = 100; u = linspace(-6, 6, gridSize);
    [A, B] = meshgrid(u, u); gridX = [A(:), B(:)];
    for j = 1:size(mu, 1)
        Z = reshape(gaussianND(gridX, mu(j, :), sigma{j}), gridSize, gridSize);
        contour(u, u, Z);
    end
    plot(mu(:, 1), mu(:, 2), 'kx', 'markersize', 12, 'linewidth', 2);
    axis([-6 6 -6 6]); title(titleStr);
end
