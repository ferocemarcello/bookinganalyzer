%/usr/local/MATLAB/R2019b/bin/matlab -nodisplay -nosplash -nodesktop -r "cd('/home/marcelloferoce/Scrivania/matlabscripts'); betantf_test_all();exit"
clear
addpath('/home/marcelloferoce/Scrivania/matlabscripts/betaNTF');
addpath('/home/marcelloferoce/Scrivania/matlabscripts');
addpath('/home/marcelloferoce/Scrivania/matlabscripts/tensorlab_2016-03-28');
% options=struct();
% options.SolverOptions=struct();
% options.SolverOptions.Compression=false;
% disp("starting rankest");
% r=rankest(tensor,options);
% disp("got rankest");
% disp(r);

%remove beach pool staff location ...
numcomp=3;
top_n=true;
top_n_toks=20;
top_n_origs=10;
top_n_dests=10;
alldata=false;
top_peaks=false;
num_high_eq_rev=0;
%for num_high_eq_rev=[0 5 10 20 30 40 50 100]:
for num_high_eq_rev=[0 5 10 20 30 40 50 100]
    disp("num_high_eq_rev= "+string(num_high_eq_rev));
    if num_high_eq_rev>0
        tens= load('/home/marcelloferoce/Scrivania/tensors/all_tensor_higher_equal_'+string(num_high_eq_rev)+'_reviews.mat');
        coi=readtable('/home/marcelloferoce/Scrivania/tensors/all_higher_equal_'+string(num_high_eq_rev)+'_reviews_new_country_origin_index.csv');
        cdi=readtable('/home/marcelloferoce/Scrivania/tensors/all_higher_equal_'+string(num_high_eq_rev)+'_reviews_new_country_destination_index.csv');
        ti = readtable('/home/marcelloferoce/Scrivania/tensors/all_higher_equal_'+string(num_high_eq_rev)+'_reviews_tokens_new_token_index.csv','HeaderLines',2);%removing token ""
    else
        tens= load('/home/marcelloferoce/Scrivania/tensors/all_tensor.mat');
        coi=readtable('/home/marcelloferoce/Scrivania/tensors/all_new_country_origin_index.csv');
        cdi=readtable('/home/marcelloferoce/Scrivania/tensors/all_new_country_destination_index.csv');
        ti = readtable('/home/marcelloferoce/Scrivania/tensors/all_tokens_new_token_index.csv','HeaderLines',2);%removing token ""
    end
    tens=tens.t;
    tens(:,:,1)=[];%removing token ""
    ti.Var1=ti.Var1-1;%removing token ""
    indices=[];
    words_to_remove=[];
    cd('/home/marcelloferoce/Scrivania');
    [num,txt,raw]=xlsread('check_tokens_all_final');
    words=cell2table(raw);
    for i=2:height(words)
        cel=words.raw5(i);
        marker=cel{1};
        if marker==0
            word=words.raw1(i);
            word=word{1};
            word=string(word(2:end-1));
            words_to_remove=[words_to_remove word];
        end
    end
    removing_indices=[];
    for tok=words_to_remove
        if sum((ti.Var2==tok))==1
            ind=find(ti.Var2==tok);
            removing_indices=[removing_indices ind];
        end
    end
    removing_indices=sort(removing_indices, 'desc');
    for i=removing_indices
        tens(:,:,i)=[];
        ti(i,:)=[];
        ti.Var1(i:height(ti))=ti.Var1(i:height(ti))-1;
    end
    addpath(genpath('/home/marcelloferoce/Scrivania/matlabscripts/tensor_toolbox'));
    for numcomp=[3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20]
        disp("numcomp= "+string(numcomp));
        cpals=cp_als_new(tensor(tens),numcomp,'printitn',0);
        lambdas_cp_als=cpals.lambda;
        max_lambda_cp_als=max(lambdas_cp_als);
        [W,H,Q,L] = betaNTF(tens,numcomp);%https://github.com/andrewssobral/lrslibrary/tree/master/algorithms/ntf/betaNTF
    %     tens(isnan(tens))=0;
    %     tens=tensor(tens);
    %    cpals=cp_als(tens,numcomp);
    %     computelambda
    %     Khatri-Rao product
        if num_high_eq_rev>0
            mkdir('/home/marcelloferoce/Scrivania/tensors/betantf/higher_equal_'+string(num_high_eq_rev)+'_reviews/peaks/together/'+string(numcomp)+'_components/');
            mkdir('/home/marcelloferoce/Scrivania/tensors/betantf/higher_equal_'+string(num_high_eq_rev)+'_reviews/top_n/together/'+string(numcomp)+'_components/');
            mkdir('/home/marcelloferoce/Scrivania/tensors/betantf/higher_equal_'+string(num_high_eq_rev)+'_reviews/top_n/together/'+string(numcomp)+'_components/normalized');
            mkdir('/home/marcelloferoce/Scrivania/tensors/betantf/higher_equal_'+string(num_high_eq_rev)+'_reviews/alldata/together/'+string(numcomp)+'_components/');
        else
            mkdir('/home/marcelloferoce/Scrivania/tensors/betantf/peaks/together/'+string(numcomp)+'_components/');
            mkdir('/home/marcelloferoce/Scrivania/tensors/betantf/top_n/together/'+string(numcomp)+'_components/');
            mkdir('/home/marcelloferoce/Scrivania/tensors/betantf/top_n/together/'+string(numcomp)+'_components/normalized/');
            mkdir('/home/marcelloferoce/Scrivania/tensors/betantf/alldata/together/'+string(numcomp)+'_components/');
        end
        maxlambda_new=0;
        for j=1:numcomp
            lambda_new=max(W(:,j))*max(H(:,j))*max(Q(:,j));
            maxlambda_new=max(maxlambda_new,lambda_new);
        end
        maxlambda_norm=0;
        for j=1:numcomp
            wh=W(:,j)*transpose(H(:,j));
            dim1=size(wh);dim2=size(Q(:,j));
            whj=reshape(reshape(wh, [], 1)*reshape(Q(:,j), 1, []), [dim1 dim2]);
            lambda_norm=sqrt(sum(whj(:).^2));
            maxlambda_norm=max(maxlambda_norm,lambda_norm);
        end
        max_lambda_old=max([max(W(:)),max(H(:)),max(Q(:))]);
        for j=1:numcomp
            disp(j);
            tab_comp_lambda_new=table();
            tab_comp_lambda_old=table();
            tab_comp_lambda_cp_als=table();
            tab_comp_lambda_norm=table();
            lambda_cp_als=lambdas_cp_als(j);
            lambda_old=max([max(W(:,j)),max(H(:,j)),max(Q(:,j))]);
            lambda_new=max(W(:,j))*max(H(:,j))*max(Q(:,j));
            
            wh=W(:,j)*transpose(H(:,j));
            dim1=size(wh);dim2=size(Q(:,j));
            whj=reshape(reshape(wh, [], 1)*reshape(Q(:,j), 1, []), [dim1 dim2]);
            lambda_norm=sqrt(sum(whj(:).^2));
            
            normalized_w=W(:,j)/lambda_new;
            normalized_h=H(:,j)/lambda_new;
            normalized_q=Q(:,j)/lambda_new;
            tab_comp_lambda_old=[tab_comp_lambda_old;{string(lambda_old)};{string(max_lambda_old)}];
            tab_comp_lambda_old.Properties.VariableNames={'lambda_old-lambda_old_max'};
            tab_comp_lambda_new=[tab_comp_lambda_new;{string(lambda_new)};{string(maxlambda_new)}];
            tab_comp_lambda_new.Properties.VariableNames={'lambda_new-lambda_new_max'};
            tab_comp_lambda_cp_als=[tab_comp_lambda_cp_als;{string(lambda_cp_als)};{string(max_lambda_cp_als)}];
            tab_comp_lambda_cp_als.Properties.VariableNames={'lambda_cp_als-lambda_cp_als_max'};
            tab_comp_lambda_norm=[tab_comp_lambda_norm;{string(lambda_norm)};{string(maxlambda_norm)}];
            tab_comp_lambda_norm.Properties.VariableNames={'lambda_norm-lambda_norm_max'};
            if alldata
                tab_comp_orig_all=table();
                tab_comp_dest_all=table();
                tab_comp_tok_all=table();
                for i=1:length(W(:,j))
                    co=string(table2cell(coi(i,2)));
                    cell={co,W(i,j)};
                    tab_comp_orig_all=[tab_comp_orig_all ;cell];
                end
                for i=1:length(H(:,j))
                    cod=string(table2cell(cdi(i,2)));
                    cell={cod,H(i,j)};
                    tab_comp_dest_all=[tab_comp_dest_all ;cell];
                end
                for i=1:length(Q(:,j))
                    to=string(table2cell(ti(i,2)));
                    cell={to,Q(i,j)};
                    tab_comp_tok_all=[tab_comp_tok_all ;cell];
                end
                maxrows_all=max([height(tab_comp_orig_all), height(tab_comp_dest_all), height(tab_comp_tok_all)]);
                if height(tab_comp_orig_all)<maxrows_all
                    for i=1:maxrows_all-height(tab_comp_orig_all)
                        tab_comp_orig_all=[tab_comp_orig_all;{"",""}];
                    end
                end
                if height(tab_comp_dest_all)<maxrows_all
                    for i=1:maxrows_all-height(tab_comp_dest_all)
                        tab_comp_dest_all=[tab_comp_dest_all;{"",""}];
                    end
                end
                if height(tab_comp_tok_all)<maxrows_all
                    for i=1:maxrows_all-height(tab_comp_tok_all)
                        tab_comp_tok_all=[tab_comp_tok_all;{"",""}];
                    end
                end
                tab_comp_orig_all.Properties.VariableNames={'country_code_origins','weight_origins'};
                tab_comp_dest_all.Properties.VariableNames={'country_code_destinations','weight_destinations'};
                tab_comp_tok_all.Properties.VariableNames={'tokens','weight_tokens'};
                together_tab_all=[tab_comp_orig_all tab_comp_dest_all tab_comp_tok_all];
                cd('/home/marcelloferoce/Scrivania/tensors/betantf/higher_equal_'+string(num_high_eq_rev)+'_reviews/alldata/together/'+string(numcomp)+'_components/');
                writetable(together_tab_all,'component_'+string(j)+'_together_components.csv','Delimiter','|','QuoteStrings',true)
            end
    %             W(:,j)=W(:,j)/norm(W(:,j));
    %             H(:,j)=H(:,j)/norm(H(:,j));
    %             Q(:,j)=Q(:,j)/norm(Q(:,j));
    %             recons_mat_comp=nan(size(tensor));
    %             two_d_mat=W(:,j).* H(:,j)';
    %             for i = 1:length(Q(:,j)) % for each level of the 3-D matrix
    %                 recons_mat_comp(:,:,i) = two_d_mat * Q(1,j);  % multiply the corresponding 2D matrices
    %             end
            if top_peaks
                tab_comp_orig_peaks=table();
                tab_comp_dest_peaks=table();
                tab_comp_tok_peaks=table();
                peaks_indices_origins=get_peaks(W(:,j),3);
                peaks_indices_destinations=get_peaks(H(:,j),3);
                peaks_indices_tokens=get_peaks(Q(:,j),3);
                for i=1:length(peaks_indices_origins)
                    ind=peaks_indices_origins(i);
                    co=string(table2cell(coi(ind,2)));
                    cell={co,W(ind,j)};
                    tab_comp_orig_peaks=[tab_comp_orig_peaks;cell];
                end
                tab_comp_orig_peaks.Properties.VariableNames = {'country_code','weight'};
                %writetable(tab_comp_orig,'component_'+string(j)+'_origins.csv','Delimiter','|','QuoteStrings',true)
                for i=1:length(peaks_indices_destinations)
                    ind=peaks_indices_destinations(i);
                    co=string(table2cell(cdi(ind,2)));
                    cell={co,H(ind,j)};
                    tab_comp_dest_peaks=[tab_comp_dest_peaks;cell];
                end
                tab_comp_dest_peaks.Properties.VariableNames = {'country_code','weight'};
                %writetable(tab_comp_dest,'component_'+string(j)+'_destinations.csv','Delimiter','|','QuoteStrings',true)
                for i=1:length(peaks_indices_tokens)
                    ind=peaks_indices_tokens(i);
                    to=string(table2cell(ti(ind,2)));
                    cell={to,Q(ind,j)};
                    tab_comp_tok_peaks=[tab_comp_tok_peaks;cell];
                end
                tab_comp_tok_peaks.Properties.VariableNames = {'token','weight'};
                %writetable(tab_comp_tok,'component_'+string(j)+'_tokens.csv','Delimiter','|','QuoteStrings',true)
                maxrows_peaks=max([height(tab_comp_orig_peaks), height(tab_comp_dest_peaks), height(tab_comp_tok_peaks)]);
                if height(tab_comp_orig_peaks)<maxrows_peaks
                    for i=1:maxrows_peaks-height(tab_comp_orig_peaks)
                        tab_comp_orig_peaks=[tab_comp_orig_peaks;{"",""}];
                    end
                end
                if height(tab_comp_dest_peaks)<maxrows_peaks
                    for i=1:maxrows_peaks-height(tab_comp_dest_peaks)
                        tab_comp_dest_peaks=[tab_comp_dest_peaks;{"",""}];
                    end
                end
                if height(tab_comp_tok_peaks)<maxrows_peaks
                    for i=1:maxrows_peaks-height(tab_comp_tok_peaks)
                        tab_comp_tok_peaks=[tab_comp_tok_peaks;{"",""}];
                    end
                end
                tab_comp_orig_peaks.Properties.VariableNames={'country_code_origins','weight_origins'};
                tab_comp_dest_peaks.Properties.VariableNames={'country_code_destinations','weight_destinations'};
                tab_comp_tok_peaks.Properties.VariableNames={'tokens','weight_tokens'};
                together_tab_peaks=[tab_comp_orig_peaks tab_comp_dest_peaks tab_comp_tok_peaks];
                cd('/home/marcelloferoce/Scrivania/tensors/betantf/higher_equal_'+string(num_high_eq_rev)+'_reviews/peaks/together/'+string(numcomp)+'_components/');
                writetable(together_tab_peaks,'component_'+string(j)+'_together_components.csv','Delimiter','|','QuoteStrings',true)
            end
            if top_n
                tab_comp_orig_top_n=table();
                tab_comp_dest_top_n=table();
                tab_comp_tok_top_n=table();
                top_n_origins=(sortrows([W(:,j) transpose(1:length(W(:,j)))],1,'descend'));
                top_n_destinations=(sortrows([H(:,j) transpose(1:length(H(:,j)))],1,'descend'));
                top_n_tokens=(sortrows([Q(:,j) transpose(1:length(Q(:,j)))],1,'descend'));
                top_n_origins=top_n_origins(1:top_n_origs,:);
                top_n_destinations=top_n_destinations(1:top_n_dests,:);
                top_n_tokens=top_n_tokens(1:top_n_toks,:);

                tab_comp_orig_top_n_normalized=table();
                tab_comp_dest_top_n_normalized=table();
                tab_comp_tok_top_n_normalized=table();
                top_n_origins_normalized=(sortrows([normalized_w transpose(1:length(normalized_w))],1,'descend'));
                top_n_destinations_normalized=(sortrows([normalized_h transpose(1:length(normalized_h))],1,'descend'));
                top_n_tokens_normalized=(sortrows([normalized_q transpose(1:length(normalized_q))],1,'descend'));
                top_n_origins_normalized=top_n_origins_normalized(1:top_n_origs,:);
                top_n_destinations_normalized=top_n_destinations_normalized(1:top_n_dests,:);
                top_n_tokens_normalized=top_n_tokens_normalized(1:top_n_toks,:);

                for i=1:length(top_n_origins)
                    ind=top_n_origins(i,2);
                    co=string(table2cell(coi(ind,2)));
                    cell={co,W(ind,j)};
                    tab_comp_orig_top_n=[tab_comp_orig_top_n;cell];
                end
                tab_comp_orig_top_n.Properties.VariableNames = {'country_code_origins','weight_origins'};
                for i=1:length(top_n_destinations)
                    ind=top_n_destinations(i,2);
                    co=string(table2cell(cdi(ind,2)));
                    cell={co,H(ind,j)};
                    tab_comp_dest_top_n=[tab_comp_dest_top_n;cell];
                end
                tab_comp_dest_top_n.Properties.VariableNames = {'country_code_destinations','weight_destinations'};
                for i=1:length(top_n_tokens)
                    ind=top_n_tokens(i,2);
                    to=string(table2cell(ti(ind,2)));
                    cell={to,Q(ind,j)};
                    tab_comp_tok_top_n=[tab_comp_tok_top_n;cell];
                end
                tab_comp_tok_top_n.Properties.VariableNames = {'tokens','weight_tokens'};
                maxrows_top_n=max([height(tab_comp_orig_top_n), height(tab_comp_dest_top_n), height(tab_comp_tok_top_n)]);

                for i=1:length(top_n_origins_normalized)
                    ind=top_n_origins_normalized(i,2);
                    co=string(table2cell(coi(ind,2)));
                    cell={co,normalized_w(ind)};
                    tab_comp_orig_top_n_normalized=[tab_comp_orig_top_n_normalized;cell];
                end
                tab_comp_orig_top_n_normalized.Properties.VariableNames = {'country_code_origins','weight_origins'};
                for i=1:length(top_n_destinations_normalized)
                    ind=top_n_destinations_normalized(i,2);
                    co=string(table2cell(cdi(ind,2)));
                    cell={co,normalized_h(ind)};
                    tab_comp_dest_top_n_normalized=[tab_comp_dest_top_n_normalized;cell];
                end
                tab_comp_dest_top_n_normalized.Properties.VariableNames = {'country_code_destinations','weight_destinations'};
                for i=1:length(top_n_tokens_normalized)
                    ind=top_n_tokens_normalized(i,2);
                    to=string(table2cell(ti(ind,2)));
                    cell={to,normalized_q(ind)};
                    tab_comp_tok_top_n_normalized=[tab_comp_tok_top_n_normalized;cell];
                end
                tab_comp_tok_top_n_normalized.Properties.VariableNames = {'tokens','weight_tokens'};
                maxrows_top_n_normalized=max([height(tab_comp_orig_top_n_normalized), height(tab_comp_dest_top_n_normalized), height(tab_comp_tok_top_n_normalized)]);

                if height(tab_comp_orig_top_n)<maxrows_top_n
                    for i=1:maxrows_top_n-height(tab_comp_orig_top_n)
                        tab_comp_orig_top_n=[tab_comp_orig_top_n;{"",""}];
                    end
                end
                if height(tab_comp_dest_top_n)<maxrows_top_n
                    for i=1:maxrows_top_n-height(tab_comp_dest_top_n)
                        tab_comp_dest_top_n=[tab_comp_dest_top_n;{"",""}];
                    end
                end
                if height(tab_comp_tok_top_n)<maxrows_top_n
                    for i=1:maxrows_top_n-height(tab_comp_tok_top_n)
                        tab_comp_tok_top_n=[tab_comp_tok_top_n;{"",""}];
                    end
                end

                if height(tab_comp_orig_top_n_normalized)<maxrows_top_n
                    for i=1:maxrows_top_n_normalized-height(tab_comp_orig_top_n_normalized)
                        tab_comp_orig_top_n_normalized=[tab_comp_orig_top_n_normalized;{"",""}];
                    end
                end
                if height(tab_comp_dest_top_n_normalized)<maxrows_top_n_normalized
                    for i=1:maxrows_top_n_normalized-height(tab_comp_dest_top_n_normalized)
                        tab_comp_dest_top_n_normalized=[tab_comp_dest_top_n_normalized;{"",""}];
                    end
                end
                if height(tab_comp_tok_top_n_normalized)<maxrows_top_n
                    for i=1:maxrows_top_n-height(tab_comp_tok_top_n_normalized)
                        tab_comp_tok_top_n_normalized=[tab_comp_tok_top_n_normalized;{"",""}];
                    end
                end

                for i=1:maxrows_top_n-height(tab_comp_lambda_new)
                        tab_comp_lambda_new=[tab_comp_lambda_new;{""}];
                end
                for i=1:maxrows_top_n-height(tab_comp_lambda_old)
                        tab_comp_lambda_old=[tab_comp_lambda_old;{""}];
                end
                for i=1:maxrows_top_n-height(tab_comp_lambda_cp_als)
                        tab_comp_lambda_cp_als=[tab_comp_lambda_cp_als;{""}];
                end
                for i=1:maxrows_top_n-height(tab_comp_lambda_norm)
                        tab_comp_lambda_norm=[tab_comp_lambda_norm;{""}];
                end
                together_tab_top_n=[tab_comp_orig_top_n tab_comp_dest_top_n tab_comp_tok_top_n tab_comp_lambda_new tab_comp_lambda_old tab_comp_lambda_cp_als tab_comp_lambda_norm];
                together_tab_top_n_normalized=[tab_comp_orig_top_n_normalized tab_comp_dest_top_n_normalized tab_comp_tok_top_n_normalized tab_comp_lambda_new tab_comp_lambda_old tab_comp_lambda_cp_als tab_comp_lambda_norm];
                if num_high_eq_rev>0 
                    cd('/home/marcelloferoce/Scrivania/tensors/betantf/higher_equal_'+string(num_high_eq_rev)+'_reviews/top_n/together/'+string(numcomp)+'_components/');
                else
                    cd('/home/marcelloferoce/Scrivania/tensors/betantf/top_n/together/'+string(numcomp)+'_components/')
                end
                if lambda_new==maxlambda_new
                    filename='component_'+string(j)+'_together_components_max.csv';
                else
                    filename='component_'+string(j)+'_together_components.csv';
                end
                writetable(together_tab_top_n,filename,'Delimiter','|','QuoteStrings',true);
                if num_high_eq_rev>0
                    cd('/home/marcelloferoce/Scrivania/tensors/betantf/higher_equal_'+string(num_high_eq_rev)+'_reviews/top_n/together/'+string(numcomp)+'_components/normalized/');
                else
                    cd('/home/marcelloferoce/Scrivania/tensors/betantf/top_n/together/'+string(numcomp)+'_components/normalized/')
                end
                writetable(together_tab_top_n_normalized,filename,'Delimiter','|','QuoteStrings',true);
            end
        end
    end
end
disp("over");    
function peaks_indices=get_peaks(init_arr,factor)
    array = abs(init_arr);
    peaks_indices=[];
    max_array = max(array);
    peaks_indices=[];
    for z=1:length(array)
        delta = max_array / array(z);
        if delta <= factor
            peaks_indices=[peaks_indices z];
        end
    end
    peaks_indices=sort(peaks_indices);
end