function main()
    %/usr/local/MATLAB/R2019b/bin/matlab -nodisplay -nosplash -nodesktop -r "cd('/home/marcelloferoce/Scrivania/matlabscripts'); tensor_writer_all();exit"
    %'/media/marcelloferoce/DATI1/pyCharmWorkspac/bookinganalyzer/'
    concept_table_path='/home/marcelloferoce/Scrivania/all_diff.csv';
    concept = readtable(concept_table_path,'HeaderLines',1);  % skips the first row of data
    concept=removevars(concept,{'Var1','Var3','Var6'});
    [t,new_country_origin_index,new_country_destination_index,new_token_index]=create_tensor(concept,-1);
    filename = '/home/marcelloferoce/Scrivania/tensors/all_tensor.mat';
    disp(size(t));
    writetable(new_country_origin_index,'/home/marcelloferoce/Scrivania/tensors/all_new_country_origin_index.csv','Delimiter','|','QuoteStrings',true)
    writetable(new_country_destination_index,'/home/marcelloferoce/Scrivania/tensors/all_new_country_destination_index.csv','Delimiter','|','QuoteStrings',true)
    writetable(new_token_index,'/home/marcelloferoce/Scrivania/tensors/all_tokens_new_token_index.csv','Delimiter','|','QuoteStrings',true)
    save(filename, 't');
    disp("over");
end
function [tensor,new_country_origin_index,new_country_destination_index,new_token_index] = create_tensor(concept,nuniquereviews)
    if nuniquereviews>0
        toDelete = concept.Var5 < nuniquereviews;
        concept(toDelete,:) = [];
    end
    todel=[];
    for i=1:height(concept)
        if isequal(concept.Var4(i),{'no_country'}) | isequal(concept.Var2(i),{'no_country'})
            todel=[todel i];
        end
    end
    concept(todel,:) = [];
    uniqueorigins=unique(concept(:,1));
    uniquuedestinations=unique(concept(:,2));
    newtokens=unique(concept(:,4));
    tensor=NaN(height(uniqueorigins),height(uniquuedestinations),height(newtokens));
    %tensor=zeros(countrysize(:,1),countrysize(:,1),tokensize(:,1));
    
    new_country_origin_index=table();
    new_country_destination_index=table();
    new_token_index=table();
    origin_indices=table();
    destination_indices=table();
    token_indices=table();
    for i=1:height(uniqueorigins)
        origin_label=string(uniqueorigins.Var2(i));
        celloriginindex = {i,origin_label};
        new_country_origin_index=[new_country_origin_index;celloriginindex];
    end
    new_country_origin_index.Properties.VariableNames = {'country_origin_index','country_origin_label'};
    for i=1:height(uniquuedestinations)
        destination_label=string(uniquuedestinations.Var4(i));
        celldestinationindex = {i,destination_label};
        new_country_destination_index=[new_country_destination_index;celldestinationindex];
    end
    new_country_destination_index.Properties.VariableNames = {'country_destination_index','country_destination_label'};
    for i=1:height(newtokens)
        token_label=string(newtokens.Var7(i));
        celltokenindex = {i,token_label};
        new_token_index=[new_token_index;celltokenindex];
    end
    new_token_index.Properties.VariableNames = {'token_index','token_label'};
    disp("rows concept= "+string(height(concept)));
    disp(datestr(now, 'dd/mm/yy-HH:MM:SS'))
    for i=1:height(concept)
        if mod(i,100000)==0
            disp(string(i)+" "+datestr(now, 'dd/mm/yy-HH:MM:SS'))
        end
        or=string(concept.Var2(i));
        ori=find(new_country_origin_index.country_origin_label==or);
        origin_indices=[origin_indices;{ori}];
        des=string(concept.Var4(i));
        desi=find(new_country_destination_index.country_destination_label==des);
        destination_indices=[destination_indices;{desi}];
        tok=string(concept.Var7(i));
        toki=find(new_token_index.token_label==tok);
        token_indices=[token_indices;{toki}];
    end
    concept=removevars(concept,{'Var2','Var4','Var7'});
    concept = addvars(concept,table2array(origin_indices),'Before','Var5');
    concept = addvars(concept,table2array(destination_indices),'Before','Var5');
    concept = addvars(concept,table2array(token_indices),'After','Var5');
    conceptmatrix=table2array(concept);
    for i = 1:length(conceptmatrix)
        if mod(i,100000)==0
            disp(string(i)+" "+datestr(now, 'dd/mm/yy-HH:MM:SS'))
        end
        o=conceptmatrix(i,1);
        d=conceptmatrix(i,2);
        t=conceptmatrix(i,4);
        f=conceptmatrix(i,5);
        tensor(o,d,t)=f;
    end
end