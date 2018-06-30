function plotData(X, y)
    figure;
    
    plot(X, y, '.b;Profit;', 'MarkerSize', 15);
    ylabel('Profit in $10,000s');
    xlabel('Population of City in 10,000s')
end
