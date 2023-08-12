function CC = CC_evaluation(image1, image2)
    r = sum(sum((image1 - mean(mean(image1))) .* (image2 - mean(mean(image2))))) / sqrt(sum(sum((image1 - mean(mean(image1))).^2)) * sum(sum((image2 - mean(mean(image2))).^2)));
    CC = r;
end
