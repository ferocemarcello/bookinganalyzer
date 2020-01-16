% Low-Rank Tensor Completion (LRTC)

clear
lowest_mse=100;
lowest_mse_nreviews=0;
lowest_rse=100;
lowest_rse_nreviews=0;
for nreviews=[0 5 10 15 20 30 40 50 60 70 80 90 100]
    cd("/home/marcelloferoce/Scrivania/matlabscripts")
    t=get_tensor("/media/marcelloferoce/DATI/pyCharmWorkspac/bookinganalyzer/", 'breakfast',nreviews);
    cd("/home/marcelloferoce/Scrivania/matlabscripts/tensor-completion-tensor-recovery-master");
    t(isnan(t))=0;
    zers=sum(sum(sum(t==0)));
    nznn=sum(sum(sum(~isnan(t) & t ~= 0)));
    tot=zers+nznn;
    disp(zers+" "+zers/tot);
    nznnindices=t ~= 0;
    newnznnindices=nznnindices;
    for i=1:length(nznnindices(:,1,1))
        for j=1:length(nznnindices(1,:,1))
            for z=1:length(nznnindices(1,1,:))
                if nznnindices(i,j,z)==1
                    if rand>=0.5
                        newnznnindices(i,j,z)=0;
                    end
                end
            end
        end
    end
    opp=t*(-1);
    newt=t;
    newt(newnznnindices== 0)=0;
    sizet=size(newt);
    r = tubalrank(newt); % tubal rank
    n1 = sizet(1,1);
    n2 = sizet(1,2);
    n3 = sizet(1,3);
    %X = tprod(randn(n1,r,n3)/n1,randn(r,n2,n3)/n2); % size: n1*n2*n3
    dr = (n1+n2-r)*r*n3;
    m = 3*dr;
    p = m/(n1*n2*n3)

    omega = find(rand(n1*n2*n3,1)<p);
    %M = zeros(n1,n2,n3);
    %M(omega) = X(omega);

    length(omega)/dr
    length(omega)/(n1*n2*n3)

    m = 3*r*(n1+n2-r)*n3+1; % number of measurements
    n = n1*n2*n3;
    A = randn(m,n)/sqrt(m);
    b = newt*t(:);
    
    tsize.n1 = n1;
    tsize.n2 = n2;
    tsize.n3 = n3;
    
    opts.DEBUG = 1;
    Xhat = lrtr_Gaussian_tnn(A,b,tsize,opts);
    trank = tubalrank(Xhat);
    RSE = norm(t(:)-Xhat(:))/norm(t(:));
    MSE=immse(t,Xhat);
    
    fracrse=RSE/(norm(opp(:)-Xhat(:))/norm(opp(:)));
    fracmse=MSE/immse(opp,Xhat);
    if fracrse<lowest_rse
        lowest_rse=fracrse;
        lowest_rse_nreviews=nreviews;
    end
    if fracmse<lowest_mse
        lowest_mse=fracmse;
        lowest_mse_nreviews=nreviews;
    end
    
    disp(nreviews);
    disp("fraction rse/rse with opposite matrix= "+fracrse);
    disp("fraction mse/mse with opposite matrix= "+fracmse);

    fprintf('\nsampling rate: %f\n', p);
    fprintf('tubal rank of the underlying tensor: %d\n',r);
    fprintf('tubal rank of the recovered tensor: %d\n', trank);
    fprintf('relative recovery error: %.8e\n', RSE);
    fprintf('MSE: %.8e\n', MSE);
end
disp(lowest_rse);
disp(lowest_rse_nreviews);
disp(lowest_mse);
disp(lowest_mse_nreviews);
disp("over")