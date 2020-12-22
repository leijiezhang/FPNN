clear; 
clc; rng(10);
%f = genpath(pwd); addpath(f);
warning('off', 'MATLAB:nearlySingularMatrix');
echo anfis off;

% Preprocess dataset and select parameters for the simulation
params_selection;

nRules = 5; 
max_rules = 30;  % for the whole dataset

dataset = dataset.generateNPartitions(runs, KFoldPartition(kfolds), NoPartition);

% Error function (MSE)
error_functions = {...
    @(d,y) norm(d - y)^2/length(d),
    @(d,y) sqrt(norm(d - y)^2/length(d)),
	@(d,y) sqrt(norm(d - y)^2/(length(d) * var(d))),
    @(d,y) 10*log10(sum((d - y).^2)/sum(d.^2)),
};

runs = 1; 

n = 1; %% simplification 

%a = 1; 
z = 1; %% simplification 

fprintf('--- RUN %i/%i ---\n', n, runs); 
    
%fis = random_fis(dataset.X, dataset.Y, 2, 'gaussmf');    

k = 1; %% simplification 
    %    for k = 1:1 % should be kfolds 
    %for k = 1:kfolds 
        [trainData, testData, ~] = dataset.getFold(k);   
%         testData_f{1} = testData; testData_f{1}.X = testData.X(:,1:feature_split);
%         testData_f{2} = testData; testData_f{2}.X = testData.X(:,feature_split+1:end);


%E = evalclusters(trainData.X,'kmeans','silhouette','klist',[5:20]);

eva = evalclusters(X,'kmeans','gap','KList',[5:40])

for i = 5:5 
    [idx(:,i), center_init(:,:,i)] = kmeans(trainData.X,i);
    [s,h] = silhouette(trainData.X,center_init(:,:,i),'Euclidean');
end

for i = 1:5
    nRules = i+1;
    [Yhat_train, Yhat_test, Kmeans_errors_train(i,1), Kmeans_errors_test(i,1), rule, H_train, H_test, beta] = ...
        FNN_main_Kmeans(trainData,testData,nRules,lambda,error_functions);
    fprintf('FNN-Kmeans training error is %f with %i rules\n', Kmeans_errors_train(i,1), nRules);
    fprintf('FNN-Kmeans testing error is %f with %i rules\n', Kmeans_errors_test(i,1), nRules);

    [Yhat_train, Yhat_test, Prototype_errors_train(i,1), Prototype_errors_test(i,1), rule, H_train, H_test, beta] = ...
        FNN_main_Prototype(trainData,testData,nRules,lambda,error_functions);
    fprintf('FNN-Prototype training error is %f with %i rules\n', Prototype_errors_train(i,1), nRules);
    fprintf('FNN-Prototype testing error is %f with %i rules\n', Prototype_errors_test(i,1), nRules);
    
    
%     [Yhat_train, Yhat_test, GM_errors_train(i,1), GM_errors_test(i,1), rule, H_train, H_test, beta] = ...
%         FNN_main_GM(trainData,testData,nRules,lambda,error_functions);
%     fprintf('FNN-GM training error is %f with %i rules\n', GM_errors_train(i,1), nRules);
%     fprintf('FNN-GM testing error is %f with %i rules\n', GM_errors_test(i,1), nRules);
    
%     [Yhat_spectral, error_spectral(i,1), rule, H_spectral, beta_spectral] = ...
%         FNN_main_spectral(trainData,nRules,lambda,error_functions);
%     fprintf('FNN-Spectral error is %f with %i rules\n', error_spectral(i,1), nRules);
%     
%     [Yhat_fcm, error_fcm(i,1), P, beta_fcm] = FNN_main_FCM(trainData,nRules,lambda,error_functions);
%     fprintf('FNN-FCM error is %f with %i rules\n', error_fcm(i,1), nRules);
    
    fprintf('--------------------------------------------------------------\n');
end

% [Yhat, min_error, errors, rule, nRules, H_best, beta_best] = ...
%     FNN_main(trainData,max_rules,lambda,error_functions);
% min_error,

clf
% figure(1)
% n_error = size(Kmeans_errors_train,1);
% plot(1:n_error,Kmeans_errors_train,'-m*','LineWidth',1.5); hold on
% plot(1:n_error,Kmeans_errors_test,'--ko','LineWidth',1.5);
% title('K-means')
% xlabel('Number of rules','fontsize',12);
% ylabel('NRMSE','fontsize',12);
% legend('Kmeans training error','Kmeans testing error');
% 
% figure(2)
% n_error = size(GM_errors_train,1);
% plot(1:n_error,GM_errors_train,'r-o','LineWidth',1.5); hold on
% plot(1:n_error,GM_errors_test,'m-*','LineWidth',1.5);
% title('GM')
% xlabel('Number of rules','fontsize',12);
% ylabel('NRMSE','fontsize',12);
% legend('GM training error','GM testing error');

figure(4)
n_error = size(Kmeans_errors_train,1);
plot(1:n_error,Kmeans_errors_train(1:n_error),'-m*','LineWidth',1.5); hold on
plot(1:n_error,Kmeans_errors_test(1:n_error),'--ko','LineWidth',1.5); hold on
plot(1:n_error,Prototype_errors_train(1:n_error),'r-o','LineWidth',1.5); hold on
plot(1:n_error,Prototype_errors_test(1:n_error),'-.bs','LineWidth',1.5);
%title('Blog N=52397, D=280')
title('kc-house N=21613, D=15')
xlabel('Number of rules','fontsize',12);
ylabel('NRMSE','fontsize',12);
legend('Kmeans training error','Kmeans testing error',...
    'Prototype training error','Prototype testing error');

% figure(2)
% plot(1:100,trainData.Y(1:100,:),'b--o','LineWidth',2); hold on
% plot(1:100,Yhat_all(1:100,:),'r*','LineWidth',2);

%[s,h] = silhouette(X,clust,'Euclidean');

function [Yhat, errors, P, beta] = FNN_main_FCM(trainData,nRules,lambda,error_functions)
                [P,center,U,obj_fcn] = Compute_P(trainData.X,nRules);
                %% run FNN solver for given rule number 
                [Yhat, beta] = FNN_solve(P, trainData.Y, lambda);
                errors = error_functions{2}(trainData.Y,Yhat);                
end

function [Yhat_train, Yhat_test, errors_train, errors_test, rule, H_train, H_test, beta] =...
    FNN_main_Prototype(trainData,testData,nRules,lambda,error_functions)
                rule = Kmeans_rule(trainData.X,nRules);
                H_train = ComputeH_P(trainData.X,rule,nRules);
                %% run FNN solver for given rule number 
                [Yhat_train, beta] = FNN_solve(H_train, trainData.Y, lambda);
                errors_train = error_functions{1}(trainData.Y,Yhat_train);
                
                H_test = ComputeH_P(testData.X,rule,nRules);
                Yhat_test = H_test * beta;
                errors_test = error_functions{1}(testData.Y,Yhat_test);
end

function [Yhat_train, Yhat_test, errors_train, errors_test, rule, H_train, H_test, beta] =...
    FNN_main_Kmeans(trainData,testData,nRules,lambda,error_functions)
                rule = Kmeans_rule(trainData.X,nRules);
                H_train = ComputeH(trainData.X,rule,nRules);
                %% run FNN solver for given rule number 
                [Yhat_train, beta] = FNN_solve(H_train, trainData.Y, lambda);
                errors_train = error_functions{1}(trainData.Y,Yhat_train);
                
                H_test = ComputeH(testData.X,rule,nRules);
                Yhat_test = H_test * beta;
                errors_test = error_functions{1}(testData.Y,Yhat_test);
end

function [Yhat_train, Yhat_test, errors_train, errors_test, rule, H_train, H_test, beta] =...
    FNN_main_GM(trainData,testData,nRules,lambda,error_functions)
                rule = GM_rule(trainData.X,nRules);
                H_train = ComputeH(trainData.X,rule,nRules);
                %% run FNN solver for given rule number 
                [Yhat_train, beta] = FNN_solve(H_train, trainData.Y, lambda);
                errors_train = error_functions{2}(trainData.Y,Yhat_train);
                
                H_test = ComputeH(testData.X,rule,nRules);
                Yhat_test = H_test * beta;
                errors_test = error_functions{2}(testData.Y,Yhat_test);
end

function [Yhat, errors, rule, H, beta] = FNN_main_spectral(trainData,nRules,lambda,error_functions)
                rule = Spectral_rule(trainData.X,nRules);
                H = ComputeH(trainData.X,rule,nRules);
                %% run FNN solver for given rule number 
                [Yhat, beta] = FNN_solve(H, trainData.Y, lambda);
                errors = error_functions{2}(trainData.Y,Yhat);
                
end

function [Yhat, min_error, errors_train, rule, nRules, H_best, beta_best] = FNN_main(trainData,max_rules,lambda,error_functions)
            for i = 1:max_rules
                rule{i} = Kmeans_rule(trainData.X,i);
                H{i} = ComputeH(trainData.X,rule{i},i);
                %% run FNN solver for each rule number 
                [Yhat(:,i), beta{i}] = FNN_solve(H{i}, trainData.Y, lambda); 
                errors_train(i, 1) = error_functions{2}(trainData.Y,Yhat(:,i)); 
            end
            [min_error, nRules] = min(errors_train);
            rule = rule{nRules};
            Yhat = Yhat(:,nRules);
            H_best = H{nRules}; 
            beta_best = beta{nRules};
end
                    
function [Yhat, boptimal] = FNN_solve(H, Y, lambda)
            [~, p] = size(H);
            boptimal = (H'*H + lambda*eye(p)) \ (H'*Y);
            %% Calculate Yhat
            Yhat = H * boptimal;
end

%% Spectral Clustering 
function rule = Spectral_rule(X,nRules)
    idx = spectralcluster(X,nRules);
    for i = 1:nRules
        center(i,:) = sum(X(idx==i,:))/length(find(idx==i)); 
    end
    %rule = struct('center', [], 'width', [], 'conq', []);
    
    inVarl=size(center,2);

    for i=1:nRules
        cind=find(idx==i)'; 
        for j=1:inVarl
           rule.center(i,j) = center(i,j);
           rule.width(i,j) = std(X(cind,j));
           rule.consq(i,j) = rand(1);
        end
    end
end

%% Gaussian Mixture 
function rule = GM_rule(X,nRules)
    options = statset('MaxIter',1000);
    GMModel_train = fitgmdist(X,nRules,'Start','plus',...
        'Options',options);
%     GMModel_train = fitgmdist(X,nRules,...
%         'CovarianceType','diagonal','SharedCovariance',false,'Start','plus','Options',options);
    inVarl=size(X,2);
    for i=1:nRules
        for j=1:inVarl
            rule.center(i,j) = GMModel_train.mu(i,j);
            rule.width(i,j) = GMModel_train.Sigma(j,j,i);
            rule.consq(i,j) = rand(1);
        end
        rule.consq(i,j+1) = rand(1);
    end
end

%% K-means  
function rule = Kmeans_rule(X,nRules)
%     [idx, center] = kmeans(X,nRules);
%     %[idx_1, center_1] = kmeans(X,nRules);
%     rule = struct('center', [], 'width', [], 'conq', []);
%    
%     inVarl=size(center,2);
% 
%     for i=1:nRules
%         cind=find(idx==i)'; 
%         for j=1:inVarl
%            rule(i).center(j)=center(i,j);
%            rule(i).width(j)=std(X(cind,j));
%         end
%     end

    [idx, center_init] = kmeans(X,nRules);
    inVarl=size(center_init,2);

    for i=1:nRules
        cind=find(idx==i)';
        for j=1:inVarl
            rule.center(i,j) = center_init(i,j);
            rule.width(i,j) = std(X(cind,j));
            rule.consq(i,j) = rand(1);
        end
        rule.consq(i,j+1) = rand(1);
    end
    
end

function [rule, eva] = GapKmeans_rule(X,Gapset)
    eva = evalclusters(X,'kmeans','gap','KList',Gapset);
    nRules = eva.OptimalK;
    [idx, center] = kmeans(X,nRules);

    rule = struct('center', [], 'width', [], 'conq', []);
   
    inVarl=size(center,2);

    for i=1:nRules
        cind=find(idx==i)'; 
        for j=1:inVarl
           rule(i).center(j)=center(i,j);
           rule(i).width(j)=std(X(cind,j));
        end 
    end
    
end

%% Gaussian multiplication 
function H = ComputeH(X,rule,nRules)

    [N, d] = size(X);
    a = zeros(nRules, N, d);

    %% make sure the sigmal value of Gauss membership function is not 0
    for ii = 1:nRules
        for jj = 1:d
            if rule.width(ii,jj) == 0 || isnan(rule.width(ii,jj))
                rule.width(ii,jj) = 1e-3;
            end
            if isnan(rule.center(ii,jj))
                rule.center(ii,jj) = 1;
            end
        end
    end
    
    %% the wrong order of center and width
%     for ii = 1:nRules
%         for jj = 1:d
%             mf = fismf(@gaussmf,[rule(ii).center(jj), rule(ii).width(jj)]);
%             a(ii, :, jj) = evalmf(mf,X(:, jj));
%         end
%     end

    for ii = 1:nRules
        for jj = 1:d
            mf = fismf(@gaussmf,[rule.width(ii,jj), rule.center(ii,jj)]);
            a(ii, :, jj) = evalmf(mf,X(:, jj));
        end
    end
    
    w = prod(a, 3); %Computes the unnormalized firing strengths
    w_hat = w./(repmat(sum(w, 1), nRules, 1)); %Normalizes the firing strengths
    w_hat(find(isnan(w_hat))) = 1/nRules;
    H = [];
    for c = 1:size(w_hat, 1)
        H = [H repmat(w_hat(c, :)', 1, d+1).*[ones(N, 1) X]]; %Computes the hidden matrix
    end
end

function [P,center,U,obj_fcn] = Compute_P(X,nRules)
    [n_sample,n_dim] = size(X);
    option_fcm = [2,100,1e-5,0];
    [center,U,obj_fcn] = fcm(X,nRules,option_fcm);
    U = U';
    P = []; % (n_sample,(n_dim+1)*nRules)
    for i = 1:nRules
        P = [P, repmat(U(:,i), 1, n_dim+1).*[ones(n_sample,1), X]];
    end
end

%% Fuzzy prototype 
function H = ComputeH_P(X,rule,nRules)
    [N, d] = size(X);
    nRules = size(rule.width,1);
    a = zeros(N,nRules);
    
    for ii = 1:nRules
        for line=1:N
            norm_X(line,ii) = norm(X(line,:)-rule.center(ii,:));
        end
    end
    sigma = sum(sum(norm_X.^2))/(nRules*N);
    
    for ii = 1:nRules
        for line=1:N
            result=exp(-norm_X(line,ii)^2/(2*sigma));
            a(line,ii) = result;
        end
    end
    
    w_hat=(a./sum(a,2))';
    
    H = [];
    for c = 1:size(w_hat, 1)
        H = [H, repmat(w_hat(c, :)', 1, d+1).*[ones(N, 1) X]]; %Computes the hidden matrix
    end
end

% option_fcm = [1.5,200,1e-5,1];
% [center,U,obj_fcn] = fcm(trainData.X,nRules,option_fcm);
% 
% [idx, center_init] = kmeans(trainData.X,nRules);

% X = data; 
% nRules = 3;
% [n_sample,n_dim] = size(X); 
% [center,U,obj_fcn] = fcm(X,nRules);
% U = U';
% P = [];
% for i = 1:nRules
%     P = [P, repmat(U(:,i), 1, n_dim+1).*[ones(n_sample,1), X]];
% end




