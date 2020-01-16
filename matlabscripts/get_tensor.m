function tensor=get_tensor(projectpath, tensorname,higheruniquereviews)
    cd(projectpath);
    tensor= load('./resources/tensors/'+convertCharsToStrings(tensorname)+'_tensor_higher_equal_'+higheruniquereviews+'.mat');
    tensor=tensor.t;
end
