%/usr/local/MATLAB/R2019b/bin/matlab -nodisplay -nosplash -nodesktop -r "cd('/home/marcelloferoce/Scrivania/matlabscripts'); test3();exit"
tensorslocations='/home/marcelloferoce/Scrivania/tensors/toptokens/';
cd(tensorslocations);
nuniquereviews = [90 100];
for key=["breakfast","location","beach","bathroom","bedroom", "internet","parking","air","coffee","transportation","cleaning"]
    cd(string(tensorslocations)+key);
	disp(key);
    mkdir ./countdistribution
    %mkdir ./probdistribution
    for topn=(10:1:50)
    	disp(topn);
        for j=nuniquereviews
            tensor= load('./'+key+'_tensor_higher_equal_'+string(j)+'_top_'+string(topn)+'_tokens.mat');
            tensor=tensor.t;
            
            filename='./countdistribution/'+key+'_tensor_higher_equal_'+string(j)+'_top_'+string(topn)+'_tokens_count.png';
            fig1=figure('visible','off');
            hcount=histogram(tensor,40,'Normalization','count');
            saveas(fig1,filename);
            
%             filename='./probdistribution/'+key+'_tensor_higher_equal_'+string(j)+'_top_'+string(topn)+'_tokens_prob.png';
%             fig2=figure('visible','off');
%             hprob=histogram(tensor,40,'Normalization','probability');
%             saveas(fig2,filename);
        end
    end
end