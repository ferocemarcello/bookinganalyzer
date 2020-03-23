cd("/home/marcelloferoce/Scrivania/matlabscripts")
%https://github.com/canyilu/tensor-completion-tensor-recovery
keywords=["breakfast","location","beach","bathroom","bedroom", "internet","parking","air","coffee","transportation","cleaning"];
nrewiews=[100,90];
for key=keywords
    for nrew=nrewiews
        cd("/home/marcelloferoce/Scrivania/matlabscripts")
        t=get_tensor("/media/marcelloferoce/DATI1/pyCharmWorkspac/bookinganalyzer/", key,nrew);
        disp(size(t));
        cd("/home/marcelloferoce/Scrivania/matlabscripts/tensor-completion-tensor-recovery-master");
        t(isnan(t))=0;
        zers=sum(sum(sum(t==0)));
        nznn=sum(sum(sum(~isnan(t) & t ~= 0)));
        tot=zers+nznn;
        disp(zers+" "+zers/tot);
        newt=t;
        knownindices=false(size(t));
        todiscoverindices=false(size(t));
        for i=1:length(t(:,1,1))
            for j=1:length(t(1,:,1))
                for z=1:length(t(1,1,:))
                    if t(i,j,z)~= 0
                        if rand>=0.5
                            newt(i,j,z)=0;
                            todiscoverindices(i,j,z)=1;
                        else
                            knownindices(i,j,z)=1;
                        end
                    end
                end
            end
        end
        opts.DEBUG = 1;
        r = tubalrank(t);
        Xhat = lrtc_tnn(newt,knownindices,opts);
        trank = tubalrank(Xhat);
        old=t(todiscoverindices);
        oldsame=t(knownindices);
        recovered=Xhat(todiscoverindices);
        recoveredsame=Xhat(knownindices);
        RSE = norm(t(:)-Xhat(:))/norm(t(:));
        MSE=immse(t,Xhat);
        %https://it.mathworks.com/matlabcentral/answers/425551-how-can-i-get-mse-and-normalized-mse-both-as-performance-function-when-fitting-feed-forward-neural-n
        MSEref = mean(mean(var(t)));
        NMSE = 1-MSE/MSEref;%https://it.mathworks.com/help/ident/ref/goodnessoffit.html

        % filename = './resources/tensors/completed/ltrc_tnn/'+key+'_tensor_higher_equal_'+nrew+'.mat';
        % save(filename, 'Xhat');
        disp(key);
        disp(nrew);
        fprintf('tubal rank of the underlying tensor: %d\n',r);
        fprintf('tubal rank of the recovered tensor: %d\n', trank);
        fprintf('relative recovery error: %.8e\n', RSE);
        fprintf('MSE: %.8e\n', MSE);
        fprintf('Normalized MSE: %.8e\n', NMSE);
    end
end