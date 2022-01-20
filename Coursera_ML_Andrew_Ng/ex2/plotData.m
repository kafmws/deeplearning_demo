function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

m=size(X)(1);
negative = [];
positive = [];
for i=1:m
    if y(i)==1
        positive=[positive;X(i,:)];
    else
        negative=[negative;X(i,:)];
    end
end

% draw one point each time cause the bug when dispaly legend
plot(positive(:,1), positive(:,2), 'k+');
plot(negative(:,1), negative(:,2), 'ko');

% =========================================================================



hold off;

end
