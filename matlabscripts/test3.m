keywords=["breakfast","location","beach","bathroom","bedroom", "internet","parking","air","coffee","transportation","cleaning"];
    for key = keywords
        for topn=(10:10:50)
            concept_table_path='/media/marcelloferoce/DATI1/bookinganalyzer/resources/bow/tourist_hotel_country_freq/diff/topntokens/'+key+'/'+key+'_top_'+string(topn)+'_tokens.csv';
            concept = readtable(concept_table_path,'HeaderLines',1);  % skips the first row of data
            todel=[];
            for i=1:height(concept)
                if isequal(concept.Var4(i),{'no_country'}) | isequal(concept.Var2(i),{'no_country'})
                    todel=[todel i];
                end
            end
            concept(todel,:) = [];
            uniqueorigins=unique(concept(:,1));
            uniquuedestinations=unique(concept(:,2));
        end
    end