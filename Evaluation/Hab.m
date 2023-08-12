function HabR=Hab(grey_matrixA,grey_matrixB,grey_level)
 
[row,column]=size(grey_matrixA);
counter = zeros(256,256);
%ͳ��ֱ��ͼ
grey_matrixA=grey_matrixA+1;
grey_matrixB=grey_matrixB+1;
for i=1:row
    for j=1:column
        indexx = grey_matrixA(i,j);
        indexy = grey_matrixB(i,j);
        counter(indexx,indexy) = counter(indexx,indexy)+1;%����ֱ��ͼ
    end
end
%����������Ϣ��
total= sum(counter(:));
index = find(counter~=0);
p = counter/total;
HabR = sum(sum(-p(index).*log2(p(index))));
        
        
