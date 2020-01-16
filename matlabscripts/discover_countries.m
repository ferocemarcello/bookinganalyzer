clc
clear
most_destinations=["fr","es","us","cn","it","tr","mx","de","th","gb"];
most_origins=["us","cn","de","gb","fr","kr","jp","ca","ru","tw"];
most_origins_expenditure=["cn","us","de","gb","fr","au","ru","ca","kr","it"];
cd('/media/marcelloferoce/DATI1/pyCharmWorkspac/bookinganalyzer/')
%keywords=["breakfast","location","beach","bathroom","bedroom", "internet","parking","air","coffee","transportation","cleaning"];
keywords=["location","beach","bathroom","bedroom", "internet","parking","air","transportation","cleaning"];
for k=keywords
    fid = fopen('./resources/bow/tourist_hotel_country_freq/'+k'+'_good.csv');
    tokens = strsplit(fgetl(fid), '|');
    fclose(fid);
    tokens=tokens(1,3:102);%first 100
    tokens_good=string(tokens);

    fid = fopen('./resources/bow/tourist_hotel_country_freq/'+k'+'_bad.csv');
    tokens = strsplit(fgetl(fid), '|');
    fclose(fid);
    tokens=tokens(1,3:102);%first 100
    tokens_bad=string(tokens);
    tokens=intersect(tokens_good,tokens_bad);
    
    for r=[100 90]
        disp(k);
        disp(r);
        tensor= load('./resources/tensors/'+k+'_tensor_higher_equal_'+r+'.mat');
        oldtensor=tensor.t;
        oldtensor(isnan(oldtensor))=0;
        knownindices=oldtensor~=0;
        unknowindices=oldtensor==0;
        tensor= load('./resources/tensors/completed/halrtc/'+k+'_tensor_higher_equal_'+r+'.mat');
        newtensor=tensor.Xhat;
        discoveredindices=zeros(size(unknowindices));
        country_origin_index=readtable('./resources/tensors/'+k+'_tensor_higher_equal_'+r+'_new_country_origin_index.csv');
        country_destination_index=readtable('./resources/tensors/'+k+'_tensor_higher_equal_'+r+'_new_country_destination_index.csv');
        token_index=readtable('./resources/tensors/'+k+'_tensor_higher_equal_'+r+'_new_token_index.csv');
        highest=0;
        
        disp(length(unknowindices(:,1,1)));
        disp("starting loops");
        discoveredcountries=table();
        for i=1:length(unknowindices(:,1,1))
            disp(i);
            for j=1:length(unknowindices(1,:,1))
                for z=1:length(unknowindices(1,1,:))
                    if unknowindices(i,j,z)==1 & newtensor(i,j,z)~=0
                        if abs(newtensor(i,j,z))>highest
                            highest=abs(newtensor(i,j,z));
                            highc=[i,j,z];
                        end
                        ori=string(table2array(country_origin_index(i,2)));
                        des=string(table2array(country_destination_index(j,2)));
                        if (ismember(ori,most_origins) | ismember(ori,most_origins_expenditure)) & ismember(des,most_destinations) & ori~=des
                            tok=string(table2array(token_index(z,2)));
                            if ismember(tok,tokens)
                                discoveredcountries=[discoveredcountries;{ori des tok newtensor(i,j,z)}];
                            end
                        end
                        discoveredindices(i,j,z)=1;
                    end
                end
            end
        end
        discoveredcountries.Properties.VariableNames = {'country_origin','country_destination','token','predicted_frequence'};
        writetable(discoveredcountries,'./resources/tensors/completed/halrtc/discovered_countries/'+k+'_nreviews_higher_equal_'+r+'.csv','Delimiter','|','QuoteStrings',true);
        disp(newtensor(highc(1),highc(2),highc(3)));
        disp(country_origin_index(highc(1),:));
        disp(country_destination_index(highc(2),:));
        disp(token_index(highc(3),:));
    end
end