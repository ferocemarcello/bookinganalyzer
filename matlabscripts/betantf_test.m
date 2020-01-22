% NTF | betaNTF | Simple beta-NTF implementation (Antoine Liutkus, 2012)
% process_video('NTF', 'betaNTF', 'dataset/demo.avi', 'output/demo_beta-NTF.avi');

% Compute a simple NTF model of 3 components
keywords=["breakfast","location","beach","bathroom","bedroom", "internet","parking","air","coffee","transportation","cleaning"];
method="ltrctnn";
topn=20;
tensorcompletpath='/home/marcelloferoce/Scrivania/tensors/completed/toptokens/'+method+'/';
betantffolder='/home/marcelloferoce/Scrivania/tensors/completed/toptokens/'+method+'/betantf/';
mkdir(betantffolder);
numcomp = 3;
cd('/home/marcelloferoce/Scrivania/matlabscripts/betaNTF')
for k=keywords
    disp(k);
    for rev=[100 90]
        tosavecomponentsdir=betantffolder+k+"_higher_equal_"+string(rev)+"_top_"+string(topn)+"/";
        mkdir(tosavecomponentsdir);
        newtensor=load(tensorcompletpath+k+'_tensor_higher_equal_'+string(rev)+'_top_'+string(topn)+'_tokens.mat');
        newtensor=newtensor.Xhat;
        [W,H,Q,L] = betaNTF(newtensor,numcomp);
        
        filename = tosavecomponentsdir+"W.mat";
        save(filename, 'W');
        
        filename = tosavecomponentsdir+"H.mat";
        save(filename, 'H');
        
        filename = tosavecomponentsdir+"Q.mat";
        save(filename, 'Q');
        
        filename = tosavecomponentsdir+"L.mat";
        save(filename, 'L');
        
        %S = (newtensor - L);
    end
end
% Reconstruct your data using:
% for j = 1:I, Vhat(:,:,j) = W * diag(Q(j,:)) * H'; end

% For reconstruction
%for i = 1:length(L(1,1,:))
%    L(:,:,i) = W * diag(Q(i,:)) * H';
%end