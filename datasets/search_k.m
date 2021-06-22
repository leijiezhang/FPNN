function [best_k]=search_k(X,method,min_rule, max_rule,threshold)

    % ʹ��k-means��silhouette��ȷ��n_rule
    if strcmp(method,'silhouette')
        result=evalclusters(X,'kmeans',method,'KList',min_rule:max_rule);
        best_k=result.OptimalK;
    % �ж��Ƿ���center��empty center,thresholdһ����0.5
    elseif strcmp(method,'fcm_nonempty')
        for n_rule=min_rule:max_rule
            [~,U]=fcm(X,n_rule,[2,100,1e-5,0]);
            
            best_k=n_rule;
            if prod(max(U,[],2)>threshold)==0
                best_k=max(2,best_k-1);
                break
            end
        end
    % ����minimal centroid distance���жϡ����������Center����̫��������Ч��
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