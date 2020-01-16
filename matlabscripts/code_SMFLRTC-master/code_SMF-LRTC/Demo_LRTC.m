%% =================================================================
% This script run SMF-LRTC-based method.
%
% More detail can be found in [1]
% [1] Yu-Bang Zheng, Ting-Zhu Huang*, Teng-Yu Ji, Xi-Le Zhao*, Tai-Xiang Jiang, and Tian-Hui Ma.
%     Low-rank tensor completion via smooth matrix factorization.
%
% Please make sure your data is in range [0, 1].
%
% Created by Yu-Bang Zheng £¨zhengyubang@163.com£©
% 12/14/2018

%% =================================================================
clc;
clear;
close all;
addpath(genpath('lib'));
addpath(genpath('data'));


%% Load initial data

for tensor_num=1
    
    switch tensor_num
        case 1
            load suzie.mat
        case 2
            load hall.mat       
    end
%%
if max(X(:))>1
    X=X/max(X(:));
end


%% Sampling with random position
sample_ratio = 0.3;
fprintf('=== The sample ratio is %4.2f ===\n', sample_ratio);
Y_tensorT = X;
Nway      = size(Y_tensorT);
known     = find(rand(prod(Nway),1)<sample_ratio);
Y_tensor0 = zeros(Nway);
Y_tensor0(known) = Y_tensorT(known);
%%
%% Use SMF-LRTC
% initialization of the parameters
rho= 0.01;
opts=[];
opts.R=[10,10,10];
opts.max_rank=[85,95,65];
opts.maxit=500;
opts.tol=2*1e-4;
alpha=[1,1,1];
alpha = alpha/sum(alpha);
opts.alpha = alpha;
opts.rho1=rho/alpha(1);
opts.rho2=rho/alpha(2);
opts.rho3=rho/alpha(3);

opts2=[];
opts2.mu=10;
opts2.rho=rho*opts2.mu/alpha(3);
opts2.beta=1000;
opts2.F_it=5;

opts5=[];
opts5.mu=100;
opts5.rho=rho*opts5.mu/alpha(3);
opts5.beta=1;
opts5.F_it=5;
%%%%%
fprintf('\n');
disp('performing SMF-LRTC ... ');
t0= tic;
[Re_SMF,A,X,Out]= inc_SMF_LRTC(Y_tensor0, known, opts, opts2, opts5);
time= toc(t0);
[psnr, ssim] = quality(Y_tensorT*255, Re_SMF*255);
fprintf('PSNR: %.3f;   SSIM: %.4f;   Time: %.2f minutes \n',psnr,ssim,time/60);
imshow3D(Re_SMF,[],1)
end
