for nreviews=[0 5 10 15 20 30 40 50 60 70 80 90 100]
    t=get_tensor("/media/marcelloferoce/DATI/pyCharmWorkspac/bookinganalyzer/", 'breakfast',nreviews);
    zers=sum(sum(sum(t==0)));
    nan=sum(sum(sum(isnan(t))));
    nznn=sum(sum(sum(~isnan(t) & t ~= 0)));
    tot=nan+nznn+zers;
    disp(tot);
    disp("nreviews= "+nreviews);
    disp(nan+" "+nan/tot);
    disp(zers+" "+zers/tot);
    disp(nznn+" "+nznn/tot);
    disp(size(t));
end
disp("over")