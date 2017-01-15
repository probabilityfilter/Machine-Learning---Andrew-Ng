function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
for i=1:K
%  temp=zeros(1,n);
%  c=0;
%  for j=1:m
%    if (idx(j,1)==i)
%      temp=temp+X(j,:);
%      c=c+1;
%    endif
%  endfor
temp_idx=idx==i; %convert the idx vector from distribution of '1 to K' to 
% '0 and 1' based on which 'i' is being worked on
temp_X=X.*temp_idx; %retain only those rows of X that have a '1' in the idx vector
  centroids(i,:)=sum(temp_X)/sum(temp_idx); %add the remaining X (rows) and divide
  % by total number of 'i' th flags
endfor
% =============================================================
end

