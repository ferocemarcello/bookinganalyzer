cd("/home/marcelloferoce/Scrivania/matlabscripts")
keywords=["breakfast","location","beach","bathroom","bedroom", "internet","parking","air","coffee","transportation","cleaning"];
nrewiews=[100,90];
clc;
%mses=[ 2.51129729e-04,2.59714218e-04,4.00369965e-04,4.41423253e-04,2.13331734e-04,2.59343505e-04,1.65838017e-04,1.86766910e-04,2.23209566e-04,2.08532769e-04,1.34689577e-04,1.14747491e-04,2.01904664e-04,2.02179227e-04,2.30587737e-04,2.12125586e-04,1.40484571e-04,1.63253432e-04,3.42930936e-04,3.19537378e-04,3.49514339e-04,3.52528988e-04];
mses=[ 5.51006298e-05, 6.04861487e-05, 1.12594782e-04, 1.20303976e-04, 6.77149602e-05, 8.03015094e-05, 3.50756328e-05, 3.79220810e-05, 3.85251619e-05, 4.26386148e-05, 3.95398845e-05, 3.41077255e-05, 5.56928050e-05, 5.53482456e-05, 1.02813774e-04, 8.76159707e-05, 5.25149733e-05, 4.93868997e-05, 7.42030749e-05, 7.86639659e-05, 7.43597941e-05, 7.67207641e-05];
i=1;
for key=keywords
    for nrew=nrewiews
        cd("/home/marcelloferoce/Scrivania/matlabscripts")
        t=get_tensor("/media/marcelloferoce/DATI1/pyCharmWorkspac/bookinganalyzer/", key,nrew);
        t(isnan(t))=0;
        MSEref = mean(mean(var(t)));
        disp(key);
        disp(nrew);
        MSE=mses(i);
        i=i+1;
        NMSE = 1-MSE/MSEref;
        fprintf('Normalized MSE: %.8e\n', NMSE);
    end
end