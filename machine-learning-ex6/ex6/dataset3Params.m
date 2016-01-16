function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_candidateArray = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_candidateArray = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

for i = 1 : size(C_candidateArray,2)
    for j = 1 : size(sigma_candidateArray,2)
        C_candidate = C_candidateArray(i);
        sigma_candidate = sigma_candidateArray(j);
        model= svmTrain(X, y, C_candidate, @(x1, x2) gaussianKernel(x1, x2, sigma_candidate));
        predictions = svmPredict(model, Xval);
        loop_score = mean(double(predictions ~= yval));
        if (~exist('stored_score', 'var') || loop_score < stored_score)
            stored_score = loop_score;
            C = C_candidate;
            sigma = sigma_candidate;
        end
    end
end



% =========================================================================

end
