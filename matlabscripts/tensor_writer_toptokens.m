function main(projectpath)
    disp(projectpath);
    cd(projectpath);
    %/usr/local/MATLAB/R2019b/bin/matlab -nodisplay -nosplash -nodesktop -r "cd('/home/marcelloferoce/Scrivania/matlabscripts'); tensor_writer('/media/marcelloferoce/DATI1/pyCharmWorkspac/bookinganalyzer/');exit"
    %'/media/marcelloferoce/DATI1/pyCharmWorkspac/bookinganalyzer/'
    mkdir ./resources/tensors/toptokens/
    fid = fopen('./booking_keywords.txt');
    keywords=[];
    while 1
        line_ex = convertCharsToStrings(fgetl(fid));
        if class(line_ex)~='string'
            break;
        end
        keywords = [keywords line_ex];
    end
    nuniquereviews = [90 100];
    for key = keywords
        newdir='./resources/tensors/toptokens/'+key+'/';
        mkdir (newdir)
        for topn=(10:1:50)
            concept_table_path='./resources/bow/tourist_hotel_country_freq/diff/topntokens/'+key+'/'+key+'_top_'+string(topn)+'_tokens.csv';
            concept = readtable(concept_table_path,'HeaderLines',1);  % skips the first row of data
            concept=removevars(concept,{'Var1','Var3','Var6','Var8','Var9'});
            try
                for j=nuniquereviews
                    disp('key: '+key);
                    disp('min number of reviews: '+string(j));
                    disp('topn tokens: '+string(topn));
                    [t,new_country_origin_index,new_country_destination_index,new_token_index]=create_tensor(concept,j);
                    filename = './resources/tensors/toptokens/'+key+'/'+key+'_tensor_higher_equal_'+j+'_top_'+topn+'_tokens.mat';
                    disp(size(t));
                    writetable(new_country_origin_index,'./resources/tensors/toptokens/'+key+'/'+key+'_tensor_higher_equal_'+j+'_new_country_origin_index.csv','Delimiter','|','QuoteStrings',true)
                    writetable(new_country_destination_index,'./resources/tensors/toptokens/'+key+'/'+key+'_tensor_higher_equal_'+j+'_new_country_destination_index.csv','Delimiter','|','QuoteStrings',true)
                    writetable(new_token_index,'./resources/tensors/toptokens/toptokens/'+key+'/'+key+'_tensor_higher_equal_'+j+'_top_'+topn+'_tokens_new_token_index.csv','Delimiter','|','QuoteStrings',true)
                    save(filename, 't');
                end
            catch e
                disp("exception");
                disp(e);
                continue
            end
        end
    end
    disp("over");
end
function [tensor,new_country_origin_index,new_country_destination_index,new_token_index] = create_tensor(concept,nuniquereviews)
    toDelete = concept.Var5 < nuniquereviews;
    concept(toDelete,:) = [];
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
    for i=1:height(concept)
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
        o=conceptmatrix(i,1);
        d=conceptmatrix(i,2);
        t=conceptmatrix(i,4);
        f=conceptmatrix(i,5);
        tensor(o,d,t)=f;
    end
end