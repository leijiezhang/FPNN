function [best_k]=search_k(X,method,min_rule, max_rule,threshold)

    % 使用k-means和silhouette来确定n_rule
    if strcmp(method,'silhouette')
        result=evalclusters(X,'kmeans',method,'KList',min_rule:max_rule);
        best_k=result.OptimalK;
    % 判断是否有center是empty center,threshold一般是0.5
    elseif strcmp(method,'fcm_nonempty')
        for n_rule=min_rule:max_rule
            [~,U]=fcm(X,n_rule,[2,100,1e-5,0]);
            
            best_k=n_rule;
            if prod(max(U,[],2)>threshold)==0
                best_k=max(2,best_k-1);
                break
            end
        end
    % 根据minimal centroid distance来判断。如果有两个Center距离太近，则无效。
    elseif strcmp(method,'fcm_MCD')
        for n_rule=min_rule:max_rule
            [center,~]=fcm(X,n_rule,[2,100,1e-5,0]);
            
            best_k=n_rule;
            if MCD_judge(center,threshold)==0
                best_k=max(2,best_k-1);
                break
            end
        end
    end
end
    
function [result]=MCD_judge(center,threshold)
    dists=pdist(center);
    result=prod(dists>=threshold);
end