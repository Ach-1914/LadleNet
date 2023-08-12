function MSE = MSE_evaluation(image1, image2)
[m, n] = size(image1);
MSE = sum(sum((image1 - image2).^2)) / (m * n);
end
