function PSNR = PSNR_evaluation(image1, image2)
[m, n] = size(image1);
MSE = sum(sum((image1 - image2).^2)) / (m * n);
PSNR = log10(255 / sqrt(MSE));
end