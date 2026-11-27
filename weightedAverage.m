function [ val ] = weightedAverage(weights, values)
% weightedAverage - average of the rows of 'values' where row i counts with weight weights(i).
%   values  - m x n matrix, one data point per row.
%   weights - m x 1 column, one weight per row (here: the responsibilities W(:,j)).
% Returns val, a 1 x n row:  val = sum_i weights(i) * values(i,:)  /  sum_i weights(i)

    % weights' is 1 x m and values is m x n, so the matrix product is the 1 x n row
    % sum_i weights(i) * values(i,:), the weighted sum of the points.
    val = weights' * values;

    % Divide by the sum of the weights (sum over the rows) so that it becomes an average.
    val = val ./ sum(weights, 1);
    
end
