function vifp=vifp_mscale(ref,dist)

sigma_nsq=2;

num=0;
den=0;
for scale=1:4
   
    N=2^(4-scale+1)+1;
    win=fspecial('gaussian',N,N/5);
    
    if (scale >1)
        ref=filter2(win,ref,'valid');
        dist=filter2(win,dist,'valid');
        ref=ref(1:2:end,1:2:end);
        dist=dist(1:2:end,1:2:end);
    end
    
    mu1   = filter2(win, ref, 'valid');
    mu2   = filter2(win, dist, 'valid');
    mu1_sq = mu1.*mu1;
    mu2_sq = mu2.*mu2;
    mu1_mu2 = mu1.*mu2;
    sigma1_sq = filter2(win, ref.*ref, 'valid') - mu1_sq;
    sigma2_sq = filter2(win, dist.*dist, 'valid') - mu2_sq;
    sigma12 = filter2(win, ref.*dist, 'valid') - mu1_mu2;
    
    sigma1_sq(sigma1_sq<0)=0;
    sigma2_sq(sigma2_sq<0)=0;
    
    g=sigma12./(sigma1_sq+1e-10);
    sv_sq=sigma2_sq-g.*sigma12;
    
    g(sigma1_sq<1e-10)=0;
    sv_sq(sigma1_sq<1e-10)=sigma2_sq(sigma1_sq<1e-10);
    sigma1_sq(sigma1_sq<1e-10)=0;
    
    g(sigma2_sq<1e-10)=0;
    sv_sq(sigma2_sq<1e-10)=0;
    
    sv_sq(g<0)=sigma2_sq(g<0);
    g(g<0)=0;
    sv_sq(sv_sq<=1e-10)=1e-10;
    
    
     num=num+sum(sum(log10(1+g.^2.*sigma1_sq./(sv_sq+sigma_nsq))));
     den=den+sum(sum(log10(1+sigma1_sq./sigma_nsq)));
    
end
vifp=num/den;