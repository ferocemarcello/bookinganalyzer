function main(projectpath)
    disp(projectpath);
    cd(projectpath);
    %/usr/local/MATLAB/R2019b/bin/matlab -nodisplay -nosplash -nodesktop -r "cd('/home/marcelloferoce/Scrivania/matlabscripts'); tensor_writer('/media/marcelloferoce/DATI1/pyCharmWorkspac/bookinganalyzer/');exit"
    %'/media/marcelloferoce/DATI1/pyCharmWorkspac/bookinganalyzer/'
    mkdir ./resources/tensors
    fid = fopen('./booking_keywords.txt');
    keywords=[];
    while 1
        line_ex = convertCharsToStrings(fgetl(fid));
        if class(line_ex)~='string'
            break;
        end
        keywords = [keywords line_ex];
    end
    nuniquereviews = [0 5 10 15 20 30 40 50 60 70 80 90 100];
    for key = keywords
        concept_table_path='./resources/bow/tourist_hotel_country_freq/diff/filtered/all_separetely/'+key+'.csv';
        country_index_table_path='./resources/bow/tourist_hotel_country_freq/diff/filtered/all_separetely/'+key+'_country_index.csv';
        token_index_table_path='./resources/bow/tourist_hotel_country_freq/diff/filtered/all_separetely/'+key+'_token_index.csv';
        try
            for j=nuniquereviews
                disp(key);
                disp(j)
                [t,new_country_origin_index,new_country_destination_index,new_token_index]=create_tensor(concept_table_path,country_index_table_path,token_index_table_path,j);
                filename = './resources/tensors/'+key+'_tensor_higher_equal_'+j+'.mat';
                disp(size(t));
                writetable(new_country_origin_index,'./resources/tensors/'+key+'_tensor_higher_equal_'+j+'_new_country_origin_index.csv','Delimiter','|','QuoteStrings',true)
                writetable(new_country_destination_index,'./resources/tensors/'+key+'_tensor_higher_equal_'+j+'_new_country_destination_index.csv','Delimiter','|','QuoteStrings',true)
                writetable(new_token_index,'./resources/tensors/'+key+'_tensor_higher_equal_'+j+'_new_token_index.csv','Delimiter','|','QuoteStrings',true)
                save(filename, 't');
            end
        catch e
            disp("exception");
            disp(e);
            continue
        end
    end
    disp("over");
end
function [tensor,new_country_origin_index,new_country_destination_index,new_token_index] = create_tensor(concept_table_path,country_index_table_path,token_index_table_path,nuniquereviews)
    concept = readtable(concept_table_path, 'HeaderLines',1);  % skips the first row of data
    country_index=readtable(country_index_table_path, 'HeaderLines',1);  % skips the first row of data
    token_index=readtable(token_index_table_path, 'HeaderLines',1);  % skips the first row of data
    toDelete = concept.Var3 < nuniquereviews;
    concept(toDelete,:) = [];
    conceptsize=size(concept);
    conceptmatrix=table2array(concept);
    uniqueorigins=unique(conceptmatrix(:,1));
    uniquuedestinations=unique(conceptmatrix(:,2));
    newcountries=unique(cat(1,uniqueorigins,uniquuedestinations));
    newtokens=unique(conceptmatrix(:,4));
    countrysize=length(newcountries);
    tokensize=length(newtokens);
    tensor=NaN(length(uniqueorigins),length(uniquuedestinations),tokensize);
    %tensor=zeros(countrysize(:,1),countrysize(:,1),tokensize(:,1));
    
    new_country_origin_index=table();
    new_country_destination_index=table();
    new_token_index=table();
    for i=1:length(uniqueorigins)
        oi=uniqueorigins(i);
        origin_label=string(country_index(oi,2).Var2);
        celloriginindex = {i,origin_label};
        new_country_origin_index=[new_country_origin_index;celloriginindex];
    end
    new_country_origin_index.Properties.VariableNames = {'country_origin_index','country_origin_label'};
    for i=1:length(uniquuedestinations)
        di=uniquuedestinations(i);
        destination_label=string(country_index(di,2).Var2);
        celldestinationindex = {i,destination_label};
        new_country_destination_index=[new_country_destination_index;celldestinationindex];
    end
    new_country_destination_index.Properties.VariableNames = {'country_destination_index','country_destination_label'};
    for i=1:length(newtokens)
        ti=newtokens(i);
        token_label=string(token_index(ti,2).Var2);
        celltokenindex = {i,token_label};
        new_token_index=[new_token_index;celltokenindex];
    end
    new_token_index.Properties.VariableNames = {'token_index','token_label'};
    for i = 1:conceptsize(1,1)
        o=conceptmatrix(i,1);
        d=conceptmatrix(i,2);
        t=conceptmatrix(i,4);
        f=conceptmatrix(i,5);
        oi=find(uniqueorigins==o);
        di=find(uniquuedestinations==d);
        ti=find(newtokens==t);
        tensor(oi,di,ti)=f;
    end
end
