most_destinations=["fr","es","us","cn","it","tr","mx","de","th","gb"];
most_origins=["us","cn","de","gb","fr","kr","jp","ca","ru","tw"];
most_origins_expenditure=["cn","us","de","gb","fr","au","ru","ca","kr","it"];

topn=20;
%method="halrtc";
%method="ltrctnn";
method="bcpf";
%method="logdet";
oldtensorpath="/home/marcelloferoce/Scrivania/tensors/toptokens/";
if method=="halrtc"
    tensorcompletedpath="/home/marcelloferoce/Scrivania/tensors/completed/toptokens/halrtc/";
end
if method=="ltrctnn"
    tensorcompletedpath="/home/marcelloferoce/Scrivania/tensors/completed/toptokens/ltrctnn/";
end
if method=="bcpf"
    tensorcompletedpath="/home/marcelloferoce/Scrivania/tensors/completed/toptokens/bcpf/";
end
if method=="logdet"
    tensorcompletedpath="/home/marcelloferoce/Scrivania/tensors/completed/toptokens/logdet/";
end
keywords=["breakfast","location","beach","bathroom","bedroom", "internet","parking","air","coffee","transportation","cleaning"];
mkdir("/home/marcelloferoce/Scrivania/tensors/completed/toptokens/"+method+"/discovered_countries/");
for k=keywords
    disp(k);
    for r=[100 90]
        oldtensorfilename = oldtensorpath+k+'/'+k+'_tensor_higher_equal_'+string(r)+'_top_'+string(topn)+'_tokens.mat';
        oldtensor=load(oldtensorfilename);
        oldtensor=oldtensor.t;
        
        knownindices=oldtensor~=0;
        unknowindices=oldtensor==0;
        discoveredindices=zeros(size(unknowindices));
        
        newtensorfilename = tensorcompletedpath+k+'_tensor_higher_equal_'+string(r)+'_top_'+string(topn)+'_tokens.mat';
        newtensor=load(newtensorfilename);
        newtensor=newtensor.Xhat;
        
        country_origin_index=readtable(oldtensorpath+k+'/'+k+'_tensor_higher_equal_'+string(r)+'_new_country_origin_index.csv');
        country_destination_index=readtable(oldtensorpath+k+'/'+k+'_tensor_higher_equal_'+string(r)+'_new_country_destination_index.csv');
        token_index=readtable(oldtensorpath+k+'/'+k+'_tensor_higher_equal_'+string(r)+'_top_'+string(topn)+'_tokens_new_token_index.csv');
        
        highest=0;
        highc=[0,0,0];
        discoveredcountries=table();
        for i=1:length(unknowindices(:,1,1))
            for j=1:length(unknowindices(1,:,1))
                for z=1:length(unknowindices(1,1,:))
                    if unknowindices(i,j,z)==1 & newtensor(i,j,z)~=0
                        if abs(newtensor(i,j,z))>highest
                            highest=abs(newtensor(i,j,z));
                            highc=[i,j,z];
                        end
                        ori=string(table2array(country_origin_index(i,2)));
                        des=string(table2array(country_destination_index(j,2)));
%                         if (ismember(ori,most_origins) | ismember(ori,most_origins_expenditure)) & ismember(des,most_destinations) & ori~=des
%                             tok=string(table2array(token_index(z,2)));
%                             discoveredcountries=[discoveredcountries;{ori des tok newtensor(i,j,z)}];
%                         end
                        tok=string(table2array(token_index(z,2)));
                        discoveredcountries=[discoveredcountries;{ori des tok newtensor(i,j,z)}];
                        discoveredindices(i,j,z)=1;
                    end
                end
            end
        end
        if height(discoveredcountries)>0
            discoveredcountries.Properties.VariableNames = {'country_origin','country_destination','token','predicted_frequence'};
            writetable(discoveredcountries,'/home/marcelloferoce/Scrivania/tensors/completed/toptokens/'+method+'/discovered_countries/'+k+'_discovered_countries_higher_equal_'+string(r)+'_top_'+string(topn)+'_tokens.csv','Delimiter','|','QuoteStrings',true);
        end
    end
end
