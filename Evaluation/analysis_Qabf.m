function output = analysis_Qabf(image1, image2)
    % model parameters
    L = 1; Tg = 0.9994; kg = -15; Dg = 0.5; Ta = 0.9879; ka = -22; Da = 0.8;
    
    % Sobel Operator
    h1 = [1 2 1; 0 0 0; -1 -2 -1]; h3 = [-1 0 1; -2 0 2; -1 0 1];
    
    SAx = conv2(image1, h3, 'same'); SAy = conv2(image1, h1, 'same');
    gA = sqrt(SAx.^2 + SAy.^2); 
    [M, N] = size(SAx); aA = zeros(M, N);
    for i = 1:M
        for j = 1:N
            if (SAx(i, j) == 0)
                aA(i, j) = pi/2;
            else
                aA(i, j) = atan(SAy(i, j) / SAx(i, j));
            end
        end
    end

    SBx = conv2(image2, h3, 'same'); SBy = conv2(image2, h1, 'same');
    gB = sqrt(SBx.^2 + SBy.^2); 
    [M, N] = size(SBx); aB = zeros(M, N);
    for i = 1:M
        for j = 1:N
            if (SBx(i, j) == 0)
                aB(i, j) = pi/2;
            else
                aB(i, j) = atan(SBy(i, j) / SBx(i, j));
            end
        end
    end
    
    deno = sum(sum(gA + gB));
    nume = sum(sum(gA)) + sum(sum(gB));
    output = nume / deno;
end
