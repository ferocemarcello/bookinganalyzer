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
        cd("/home/marcelloferoce/Scrivania/matlabscripts/MF_TV-master/");
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
        X=t;
        %% Set sampling ratio
        sr = 0.5;

        %% Initial data

        % normalized data
        if max(X(:))>1
        X=X/max(X(:));
        end

        Nway=[size(X,1), size(X,2), size(X,3)];
        n1 = size(X,1); n2 = size(X,2); 
        frames=size(X,3);
        ratio = 0.005;
        cd("/home/marcelloferoce/Scrivania/matlabscripts/MF_TV-master/lib/");
        R=AdapN_Rank(X,ratio);
        Y_tensorT = X;

        p = round(sr*prod(Nway));
        known = randsample(prod(Nway),p);
        data = Y_tensorT(known);
        [known, id]= sort(known); data= data(id);
        Y_tensor0= zeros(Nway); Y_tensor0(known)= data;
        %imname=[num2str(tensor_num),'_tensor0'];
        %% Initialization of the factor matrices X and A
        for n = 1:3
            coNway(n) = prod(Nway)/Nway(n);
        end
        for i = 1:3
            Y0{i} = Unfold(Y_tensor0,Nway,i);
            Y0{i} = Y0{i}';
            X0{i}= rand(coNway(i), R(i));
            A0{i}= rand(R(i),Nway(i));
        end
        cd("/home/marcelloferoce/Scrivania/matlabscripts/MF_TV-master/");
        opts=[];
        opts.maxit=500;
        opts.Ytr= Y_tensorT;
        opts.tol=1e-5;
        alpha=[1,1,1];
        opts.alpha = alpha / sum(alpha);
        rho=0.1;
        opts.rho1=rho;
        opts.rho2=rho;
        opts.rho3=rho;
        opts.mu=1; 
        opts.beta=10;
        opts.initer=10;
        opts.miter=20;

        [Y_TV, A, X, Out]= LRTC_TV(Y0, data, A0, X0,Y_tensor0, Nway, known, opts, n1, n2);

        RSE = norm(t(:)-Y_TV(:))/norm(t(:));
        MSE=immse(t,Y_TV);
        MSEref = mean(mean(var(t)));
        NMSE = 1-MSE/MSEref;%https://it.mathworks.com/help/ident/ref/goodnessoffit.html

        % filename = './resources/tensors/completed/logdet/'+key+'_tensor_higher_equal_'+nrew+'.mat';
        % save(filename, 'Y_TV');
        disp(key);
        disp(nrew);
        fprintf('relative recovery error: %.8e\n', RSE);
        fprintf('MSE: %.8e\n', MSE);
        fprintf('Normalized MSE: %.8e\n', NMSE);
    end
end