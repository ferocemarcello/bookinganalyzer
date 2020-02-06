tensorpath="/home/marcelloferoce/Scrivania/tensors/completed";
cd(tensorpath);
mkdir("./toptokens/halrtc/");
mkdir("./toptokens/logdet/");
mkdir("./toptokens/ltrctnn/");
mkdir("./toptokens/bcpf/");
keywords=["breakfast","location","beach","bathroom","bedroom", "internet","parking","air","coffee","transportation","cleaning"];
nuniquereviews = [90 100];
topn=30;
%method="halrtc";
%method="logdet";
%method="ltrctnn";
method="bcpf";
if method=="halrtc"
    tensorcompletedpath=tensorpath+"/toptokens/halrtc/";
    cd("/home/marcelloferoce/Scrivania/matlabscripts/mctc4bmi-master/algs_tc/LRTC/");
end
if method=="logdet"
    tensorcompletedpath=tensorpath+"/toptokens/logdet/";
    cd("/home/marcelloferoce/Scrivania/matlabscripts/MF_TV-master/");
end
if method=="ltrctnn"
    tensorcompletedpath=tensorpath+"/toptokens/ltrctnn/";
    cd("/home/marcelloferoce/Scrivania/matlabscripts/tensor-completion-tensor-recovery-master");
end
if method=="bcpf"
    cd("/home/marcelloferoce/Scrivania/matlabscripts/tensor_toolbox");
    addpath(pwd) %<-- Add the tensor toolbox to the MATLAB path
    cd met; addpath(pwd) %<-- Also add the met directory
    cd("/home/marcelloferoce/Scrivania/matlabscripts/tensor-completion-tensor-recovery-master");
    addpath(pwd) %for tubalrank
    tensorcompletedpath=tensorpath+"/toptokens/bcpf/";
    cd("/home/marcelloferoce/Scrivania/matlabscripts/mctc4bmi-master/algs_tc/BCPF/Algorithms/");
end
for key=keywords
    for nrew=nuniquereviews
        
        disp(key);
        disp(nrew);
        
        tensor= load('/home/marcelloferoce/Scrivania/tensors/toptokens/'+key+'/'+key+'_tensor_higher_equal_'+string(nrew)+'_top_'+string(topn)+'_tokens.mat');
        tensor=tensor.t;
        
        tensor(isnan(tensor))=0;
        
        if method=="halrtc"
            Xhat=completehalrtc(tensor);
        end
        if method=="logdet"
            Xhat=completelogdet(tensor);
        end
        if method=="ltrctnn"
            Xhat=completeltrctnn(tensor);
        end
        if method=="bcpf"
            Xhat=completebcpf(tensor);
        end
        filename = tensorcompletedpath+key+'_tensor_higher_equal_'+string(nrew)+'_top_'+string(topn)+'_tokens.mat';
        save(filename, 'Xhat');
    end
end
function Xhat=completehalrtc(tensor)
    alpha = [1, 1, 1];
    alpha = alpha / sum(alpha);
    maxIter = 500;
    epsilon = 1e-5;
    beta = 0.8;
    knownindices=tensor~=0;

    Xhat = HaLRTC(tensor,knownindices,alpha,beta,maxIter,epsilon);
end
function Xhat=completelogdet(tensor)
        X=tensor;
        knownindices=X~=0;
        known=find(X~=0);
        %% Set sampling ratio
        sr = 0.5;

        %% Initial data

        % normalized data
%         if max(X(:))>1
%         X=X/max(X(:));
%         end

        Nway=[size(X,1), size(X,2), size(X,3)];
        n1 = size(X,1); n2 = size(X,2); 
        frames=size(X,3);
        ratio = 0.005;
        cd("./lib");
        R=AdapN_Rank(X,ratio);
        Y_tensorT = X;
        
        data = Y_tensorT(known);
        [known, id]= sort(known);
        data= data(id);
        Y_tensor0= zeros(Nway);
        Y_tensor0(known)= data;
        for n = 1:3
            coNway(n) = prod(Nway)/Nway(n);
        end
        for i = 1:3
            Y0{i} = Unfold(Y_tensor0,Nway,i);
            Y0{i} = Y0{i}';
            X0{i}= rand(coNway(i), R(i));
            A0{i}= rand(R(i),Nway(i));
        end
        cd("../");
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
        Xhat=Y_TV;
end
function Xhat=completeltrctnn(tensor)
    opts.DEBUG = 1;
    knownindices=tensor~=0;
    Xhat = lrtc_tnn(tensor,knownindices,opts);
end
function Xhat=completebcpf(tensor)
    opts.DEBUG = 1;
    alpha = [1, 1, 1e-3];
    alpha = alpha / sum(alpha);
    maxIter = 500;
    epsilon = 1e-5;
    beta = 0.1*ones(1, ndims(tensor));
    r = tubalrank(tensor);
    X=BCPF_TC(tensor, 'maxRank', r+10, 'maxiters', 500);
    Xhat=double(X.X);
end