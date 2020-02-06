%/usr/local/MATLAB/R2019b/bin/matlab -nodisplay -nosplash -nodesktop -r "cd('/home/marcelloferoce/Scrivania/matlabscripts'); betantf_test_all();exit"
clear
tensor= load('/home/marcelloferoce/Scrivania/tensors/all_tensor.mat');
tensor=tensor.t;
tensor(:,:,1)=[];%removing token ""
addpath('/home/marcelloferoce/Scrivania/matlabscripts/betaNTF');
% addpath('/home/marcelloferoce/Scrivania/matlabscripts/tensorlab_2016-03-28');
% options=struct();
% options.SolverOptions=struct();
% options.SolverOptions.Compression=false;
% disp("starting rankest");
% r=rankest(tensor,options);
% disp("got rankest");
% disp(r);
numcomp=3;
higher_10=0;
coi=readtable('/home/marcelloferoce/Scrivania/tensors/all_new_country_origin_index.csv');
cdi=readtable('/home/marcelloferoce/Scrivania/tensors/all_new_country_destination_index.csv');
ti = readtable('/home/marcelloferoce/Scrivania/tensors/all_tokens_new_token_index.csv','HeaderLines',2);%removing token ""
ti.Var1=ti.Var1-1;%removing token ""
peaks=true;
if peaks
    mkdir('/home/marcelloferoce/Scrivania/tensors/betantf/peaks/');
    for numcomp=[3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20]
        disp("numcomp= "+string(numcomp));
        [W,H,Q,L] = betaNTF(tensor,numcomp);%https://github.com/andrewssobral/lrslibrary/tree/master/algorithms/ntf/betaNTF
        mkdir('/home/marcelloferoce/Scrivania/tensors/betantf/peaks/'+string(numcomp)+'_components/');
        cd('/home/marcelloferoce/Scrivania/tensors/betantf/peaks/'+string(numcomp)+'_components/');
        for j=1:numcomp
            tab_comp_orig=table();
            tab_comp_dest=table();
            tab_comp_tok=table();
            disp(j);
            peaks_indices_origins=get_peaks(W(:,j),3);
            disp(length(peaks_indices_origins));
            peaks_indices_destinations=get_peaks(H(:,j),3);
            disp(length(peaks_indices_destinations));
            peaks_indices_tokens=get_peaks(Q(:,j),3);
            disp(length(peaks_indices_tokens));
            for i=1:length(peaks_indices_origins)
                ind=peaks_indices_origins(i);
                co=string(table2cell(coi(ind,2)));
                cell={co,W(ind,j)};
                tab_comp_orig=[tab_comp_orig;cell];
            end
            tab_comp_orig.Properties.VariableNames = {'country_code','weight'};
            writetable(tab_comp_orig,'component_'+string(j)+'_origins.csv','Delimiter','|','QuoteStrings',true)
            for i=1:length(peaks_indices_destinations)
                ind=peaks_indices_destinations(i);
                co=string(table2cell(cdi(ind,2)));
                cell={co,H(ind,j)};
                tab_comp_dest=[tab_comp_dest;cell];
            end
            tab_comp_dest.Properties.VariableNames = {'country_code','weight'};
            writetable(tab_comp_dest,'component_'+string(j)+'_destinations.csv','Delimiter','|','QuoteStrings',true)
            for i=1:length(peaks_indices_tokens)
                ind=peaks_indices_tokens(i);
                to=string(table2cell(ti(ind,2)));
                cell={to,Q(ind,j)};
                tab_comp_tok=[tab_comp_tok;cell];
            end
            tab_comp_tok.Properties.VariableNames = {'token','weight'};
            writetable(tab_comp_tok,'component_'+string(j)+'_tokens.csv','Delimiter','|','QuoteStrings',true)
            if length(peaks_indices_tokens)>10
                higher_10=higher_10+1;
            end
            if length(peaks_indices_destinations)>10
                higher_10=higher_10+1;
            end
            if length(peaks_indices_origins)>10
                higher_10=higher_10+1;
            end
        end
    end
else
    mkdir('/home/marcelloferoce/Scrivania/tensors/betantf/alldata/');
    for numcomp=[3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20]
        disp("numcomp= "+string(numcomp));
        [W,H,Q,L] = betaNTF(tensor,numcomp);%https://github.com/andrewssobral/lrslibrary/tree/master/algorithms/ntf/betaNTF
        mkdir('/home/marcelloferoce/Scrivania/tensors/betantf/alldata/'+string(numcomp)+'_components/');
        s='/home/marcelloferoce/Scrivania/tensors/betantf/alldata/'+string(numcomp)+'_components/';
        cd(s);
        for j=1:numcomp
            tab_comp_origin=table();
            tab_comp_destination=table();
            tab_comp_tokens=table();
            for i=1:length(W(:,j))
                co=string(table2cell(coi(i,2)));
                cell={co,W(i,j)};
                tab_comp_origin=[tab_comp_origin ;cell];
            end
            for i=1:length(H(:,j))
                cod=string(table2cell(cdi(i,2)));
                cell={cd,H(i,j)};
                tab_comp_destination=[tab_comp_destination ;cell];
            end
            for i=1:length(Q(:,j))
                to=string(table2cell(ti(i,2)));
                cell={to,Q(i,j)};
                tab_comp_tokens=[tab_comp_tokens ;cell];
            end
            tab_comp_origin.Properties.VariableNames = {'country_code_origin','weight'};
            tab_comp_destination.Properties.VariableNames = {'country_code_destination','weight'};
            tab_comp_tokens.Properties.VariableNames = {'token','weight'};
            writetable(tab_comp_origin,'component_'+string(j)+'_origins.csv','Delimiter','|','QuoteStrings',true)
            writetable(tab_comp_destination,'component_'+string(j)+'_destinations.csv','Delimiter','|','QuoteStrings',true)
            writetable(tab_comp_tokens,'component_'+string(j)+'_tokens.csv','Delimiter','|','QuoteStrings',true)
        end
    end
end
disp(higher_10);
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