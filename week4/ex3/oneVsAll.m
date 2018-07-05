function [allThetas] = oneVsAll(X, y, labelsCount, lambda)
    examplesCount = size(X, 1);
    featuresCount = size(X, 2);

    allThetas = zeros(labelsCount, featuresCount + 1);

    % Add ones to the X data matrix
    X = [ones(examplesCount, 1) X];

    for label = 1:labelsCount
        % Set Initial theta
        initial_theta = zeros(featuresCount + 1, 1);
        
        % Set options for fminunc
        options = optimset('GradObj', 'on', 'MaxIter', 50);

        % Run fmincg to obtain the optimal theta
        % This function will return theta and the cost 
        theta = fmincg(@(t)(lrCostFunction(t, X, (y == label), lambda)), initial_theta, options);
        allThetas(label, :) = theta';
    end
end
