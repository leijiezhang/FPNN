n=size(X,1);
idx=randperm(n);
X=X(idx,:);
Y= Y(idx,:);
n_cls = size(unique(Y),1);
x_tmp = [];
y_tmp = [];
for i=0:n_cls-1
    idx_tmp = find(Y==i);
    x_tmp = [x_tmp; X(idx_tmp(1:10),:)];
    y_tmp = [y_tmp; Y(idx_tmp(1:10),:)];
end
X=x_tmp;
Y=y_tmp;
name='esr_10';
save esr_10.mat X Y task name

