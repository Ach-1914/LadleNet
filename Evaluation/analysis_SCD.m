function r = analysis_SCD(image1, image2)
    r = corr2(image2 - image1, image1) + corr2(image1 - image2, image2);
end
