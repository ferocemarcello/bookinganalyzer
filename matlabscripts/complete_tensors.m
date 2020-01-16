cd("/home/marcelloferoce/Scrivania/matlabscripts")
keywords=["breakfast","location","beach","bathroom","bedroom", "internet","parking","air","coffee","transportation","cleaning"];
nrewiews=[100,90];
for key=keywords
    for nrew=nrewiews
        cd("/home/marcelloferoce/Scrivania/matlabscripts")
        t=get_tensor("/media/marcelloferoce/DATI1/pyCharmWorkspac/bookinganalyzer/", key,nrew);
        cd("/home/marcelloferoce/Scrivania/matlabscripts/tensor_toolbox");
        addpath(pwd) %<-- Add the tensor toolbox to the MATLAB path
        cd met; addpath(pwd) %<-- Also add the met directory
        cd("/home/marcelloferoce/Scrivania/matlabscripts/mctc4bmi-master/algs_tc/LRTC/");
        t(isnan(t))=0;
        zers=sum(sum(sum(t==0)));
        nznn=sum(sum(sum(~isnan(t) & t ~= 0)));
        tot=zers+nznn;
        disp(zers+" "+zers/tot);
        opts.DEBUG = 1;
        alpha = [1, 1, 1];
        alpha = alpha / sum(alpha);
        maxIter = 500;
        epsilon = 1e-5;
        %beta = 0.1*ones(1, ndims(t));
        beta = 0.8;
        disp(key);
        disp(nrew);
        knownindices=t~=0;
        Xhat = HaLRTC(t,knownindices,alpha,beta,maxIter,epsilon);
        cd("/media/marcelloferoce/DATI1/pyCharmWorkspac/bookinganalyzer/");
        filename = './resources/tensors/completed/halrtc/'+key+'_tensor_higher_equal_'+nrew+'.mat';
        save(filename, 'Xhat');
    end
end