most_destinations=["fr","es","us","cn","it","tr","mx","de","th","gb"];
most_origins=["us","cn","de","gb","fr","kr","jp","ca","ru","tw"];
most_origins_expenditure=["cn","us","de","gb","fr","au","ru","ca","kr","it"];

imf_report = readtable('imf-dm-export-20200125.xls') ;
topdevcountries=table();
topdevcountries=[topdevcountries imf_report.RealGDPGrowth_AnnualPercentChange_ imf_report.x2020, imf_report.x2021 imf_report.x2022 imf_report.x2023 imf_report.x2024];
topdevcountries=topdevcountries(2:end-34,:);
topdevcountries=[topdevcountries{:,1} num2cell(str2double(topdevcountries{:,2})) num2cell(str2double(topdevcountries{:,3})) num2cell(str2double(topdevcountries{:,4})) num2cell(str2double(topdevcountries{:,5})) num2cell(str2double(topdevcountries{:,6}))];
topdevcountries_tab=table();
topdevcountries_tab=[topdevcountries_tab topdevcountries(:,1) topdevcountries(:,2) topdevcountries(:,3) topdevcountries(:,4) topdevcountries(:,5) topdevcountries(:,6)];
topdevcountries_tab.Properties.VariableNames = {'country' '2020' '2021' '2022' '2023' '2024'};
mean_gdp=table(mean(topdevcountries_tab{:,2:end},2));
topdevcountries_tab=addvars(topdevcountries_tab,table2array(mean_gdp),'After','2024');
topdevcountries_tab.Properties.VariableNames(7)={'avg_gdp'};
topdevcountries_tab=sortrows(topdevcountries_tab,7,'descend');
topdevcountries_tab(find(isnan(topdevcountries_tab.avg_gdp)),:)=[];
top_countries=topdevcountries_tab(1:50,1).country;
top_countries_iso=[];
country_to_iso=readtable('country_to_iso.csv') ;
for i=1:length(top_countries)
    country=string(top_countries(i));
    if country=="Lao P.D.R."
        country="Lao People's Democratic Republic";
    end
    if country=="Vietnam"
        country="Viet Nam";
    end
    if country=="Tanzania"
        country="Tanzania, United Republic of";
    end
    if country=="South Sudan, Republic of"
        country="South Sudan";
    end
    if country=="China, People's Republic of"
        country="China";
    end
    if country=="Gambia, The"
        country="Gambia";
    end
    if country=="Cabo Verde"
        country="Cape Verde";
    end
    top_countries_iso=[top_countries_iso; lower(country_to_iso.Code(strcmp(country_to_iso.Name,country)))];
    disp(length(top_countries_iso))
end
topn=10;
method="halrtc";
%method="ltrctnn";
%method="bcpf";
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
                        if ismember(ori,top_countries_iso)
                            relevant=true;
                        else
                            relevant=false;
                        end
                        tok=string(table2array(token_index(z,2)));
                        if relevant
                            discoveredcountries=[discoveredcountries;{ori des tok newtensor(i,j,z) "relevant_country_of_origin"}];
                        else
                            discoveredcountries=[discoveredcountries;{ori des tok newtensor(i,j,z)} {""}];
                        end
                        discoveredindices(i,j,z)=1;
                    end
                end
            end
        end
        if height(discoveredcountries)>0
            discoveredcountries.Properties.VariableNames = {'country_origin','country_destination','token','predicted_frequence','relevant_origin'};
            writetable(discoveredcountries,'/home/marcelloferoce/Scrivania/tensors/completed/toptokens/'+method+'/discovered_countries/'+k+'_discovered_countries_higher_equal_'+string(r)+'_top_'+string(topn)+'_tokens.csv','Delimiter','|','QuoteStrings',true);
        end
    end
end
