clear
keywords=["breakfast","location","beach","bathroom","bedroom", "internet","parking","air","coffee","transportation","cleaning"];
%keywords="coffee";
%method="ltrctnn";
method="logdet";
topn=20;
tensorcompletedpath='/home/marcelloferoce/Scrivania/tensors/completed/toptokens/'+method+'/';
tensorincompletepath='/home/marcelloferoce/Scrivania/tensors/toptokens/';
betantffolder='/home/marcelloferoce/Scrivania/tensors/toptokens/betantf/';
ncpfolder='/home/marcelloferoce/Scrivania/tensors/toptokens/ncpfolder/';
%mkdir(betantffolder);

addpath('/home/marcelloferoce/Scrivania/matlabscripts/betaNTF');
addpath('/home/marcelloferoce/Scrivania/matlabscripts/nonnegfac-matlab-master');
addpath(genpath('/home/marcelloferoce/Scrivania/matlabscripts/tensor_toolbox'));
addpath('/home/marcelloferoce/Scrivania/matlabscripts/tensorlab_2016-03-28');
try_components=[3 5 7 10 15 20];
options.Compression=false;
for k=keywords
    disp(k);
    for rev=[100 90]
        disp(rev);
        incompletetensor=load(tensorincompletepath+k+'/'+k+'_tensor_higher_equal_'+string(rev)+'_top_'+string(topn)+'_tokens.mat');
        incompletetensor=incompletetensor.t;
        formattedtensor=fmt(incompletetensor);
         for numcomp=try_components
            tosavecomponentsdir=betantffolder+k+"_higher_equal_"+string(rev)+"_top_"+string(topn)+"/";
            %mkdir(tosavecomponentsdir);
            tosavecomponentsdir=ncpfolder+k+"_higher_equal_"+string(rev)+"_top_"+string(topn)+"/";
            %mkdir(tosavecomponentsdir);
            
            disp("num_components= "+numcomp);
            for i=1:10
                [factors, output]=cpd(formattedtensor, numcomp,options);%https://www.tensorlab.net/doc/cpd.html
                origins=cell2mat(factors(1));
                destinations=cell2mat(factors(2));
                tokens=cell2mat(factors(3));
                stds=[];
                avglen_origin_peaks=0;
                avglen_destin_peaks=0;
                avglen_token_peaks=0;
                for j=1:numcomp
                    [pks_cpd_origins,locs_cpd_origins] = findpeaks(abs(origins(:,j)));
                    peaks_indices_origins=get_peaks(origins(:,j),3);
                    avglen_origin_peaks=avglen_origin_peaks+length(peaks_indices_origins);
                    
                    [pks_cpd_destinations,locs_cpd_destinations] = findpeaks(abs(destinations(:,j)));
                    peaks_indices_destinations=get_peaks(destinations(:,j),3);
                    avglen_destin_peaks=avglen_destin_peaks+length(peaks_indices_destinations);
                    
                    [pks_cpd_tokens,locs_cpd_tokens] = findpeaks(abs(tokens(:,3)));
                    peaks_indices_tokens=get_peaks(tokens(:,j),10);
                    avglen_token_peaks=avglen_token_peaks+length(peaks_indices_tokens);
                end
            end
            avglen_origin_peaks=avglen_origin_peaks/i;
            avglen_destin_peaks=avglen_destin_peaks/i;
            avglen_token_peaks=avglen_token_peaks/i;
            disp("locations peaks cpd origins= ");
            sprintf('%d ',intersect_origins)
            disp("number of peaks cpd origins= "+string(length(intersect_origins)));
            disp("locations peaks cpd destinations= ");
            sprintf('%d ',intersect_destinations)
            disp("number of peaks cpd destinations= "+string(length(intersect_destinations)));
            disp("locations peaks cpd tokens= ");
            sprintf('%d ',intersect_tokens)
            disp("number of peaks cpd tokens= "+string(length(intersect_tokens)));
            
            [W_inc,H_inc,Q_inc,L_inc] = betaNTF(incompletetensor,numcomp);%https://github.com/andrewssobral/lrslibrary/tree/master/algorithms/ntf/betaNTF
            [pks_bntf,locs_bntf] = findpeaks(abs(W_inc(:,1)));
            disp("number of peaks beta ntf= "+string(length(locs_bntf)));
         end
    end
end
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