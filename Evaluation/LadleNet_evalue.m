clc
clear all

path1 = 'Your Path';
path2 = 'Your Path';

fileList1 = dir(fullfile(path1, '*.jpg'));
fileList2 = dir(fullfile(path2, '*.jpg'));

numImages = numel(fileList1);
EN = zeros(numImages, 1);
SF = zeros(numImages, 1);
SD = zeros(numImages, 1);
PSNR = zeros(numImages, 1);
MSE = zeros(numImages, 1);
MI = zeros(numImages, 1);
VIF = zeros(numImages, 1);
AG = zeros(numImages, 1);
CC = zeros(numImages, 1);
SCD = zeros(numImages, 1);
Qabf = zeros(numImages, 1);

for i = 1:numImages
    img1 = imread(fullfile(path1, fileList1(i).name));
    img1 = rgb2gray(img1);
    
    img2 = imread(fullfile(path2, fileList2(i).name));
    img2 = rgb2gray(img2);
    
    EN(i) = entropy(img1);
    SF(i) = SF_evaluation(img1);
    SD(i) = SD_evaluation(img1);
    PSNR(i) = PSNR_evaluation(img1, img2);
    MSE(i) = MSE_evaluation(img1, img2);
    MI(i) = MI_evaluation(img1, img2, 256);
    VIF(i) = vifp_mscale(img1, img2);
    AG(i) = AG_evaluation(img1);
    CC(i) = CC_evaluation(img1, img2);
    SCD(i) = analysis_SCD(img1, img2);
    Qabf(i) = analysis_Qabf(img1, img2);
end


headers = {'File Name', 'EN', 'SF', 'SD', 'PSNR', 'MSE', 'MI', 'VIF', 'AG', 'CC', 'SCD', 'Qabf'};
fileNames = {fileList1.name}';

data = num2cell([EN, SF, SD, PSNR, MSE, MI, VIF, AG, CC, SCD, Qabf]);

resultData = [fileNames, data];

excelFilePath = 'Your Path';
xlswrite(excelFilePath, [headers; resultData], 'Sheet1');
disp('Done!');