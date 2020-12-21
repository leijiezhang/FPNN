noise_list = ["0.0", "0.1", "0.3", "0.5"];
algorithm_list = ["SVM", "MLP21", "MLP121", "MLP212", "MLP421", "MLP12421", "MLP42124", "CNN11", "CNN12", "CNN21", "CNN22", "FPN"];
test_acc_mean = zeros(size(noise_list, 2), size(algorithm_list, 2))';
test_acc_std = zeros(size(noise_list, 2), size(algorithm_list, 2))';
for i=1:4
    load_name = sprintf("acc_fpn_esr_rule50_nl_%s_epoch_500_all.mat", noise_list(i)); 
    load(load_name);
    svm_test_acc = double(svm_test_acc_tsr);
    clear svm_test_acc_tsr;
    svm_test_acc_mean = mean(svm_test_acc)*100;
    svm_test_acc_std = std(svm_test_acc)*100;
    svm_train_acc = double(svm_train_acc_tsr);
    clear svm_train_acc_tsr
    svm_train_acc_mean = mean(svm_train_acc)*100;
    svm_train_acc_std = std(svm_train_acc)*100;
    test_acc_mean(1, i) = svm_test_acc_mean;
    test_acc_std(1, i) = svm_test_acc_std;
    
%     fprintf("svm results %s: & %.2f/%.2f  & %.2f/%.2f \n",...
%             noise_list(i), svm_train_acc_mean, svm_train_acc_std, svm_test_acc_mean, svm_test_acc_std);
    mlp21_test_acc = double(mlp21_test_acc_tsr(end, :));
    clear mlp21_test_acc_tsr;
    mlp21_test_acc_mean = mean(mlp21_test_acc)*100;
    mlp21_test_acc_std = std(mlp21_test_acc)*100;
    mlp21_train_acc = double(mlp21_train_acc_tsr(end, :));
    clear mlp21_train_acc_tsr
    mlp21_train_acc_mean = mean(mlp21_train_acc)*100;
    mlp21_train_acc_std = std(mlp21_train_acc)*100;
    test_acc_mean(2, i) = mlp21_test_acc_mean;
    test_acc_std(2, i) = mlp21_test_acc_std;
%     fprintf("mlp21 results %s: & %.2f/%.2f  & %.2f/%.2f \n",...
%             noise_list(i), mlp21_train_acc_mean, mlp21_train_acc_std, mlp21_test_acc_mean, mlp21_test_acc_std);

    mlp121_test_acc = double(mlp121_test_acc_tsr(end, :));
    clear mlp121_test_acc_tsr;
    mlp121_test_acc_mean = mean(mlp121_test_acc)*100;
    mlp121_test_acc_std = std(mlp121_test_acc)*100;
    mlp121_train_acc = double(mlp121_train_acc_tsr(end, :));
    clear mlp121_train_acc_tsr
    mlp121_train_acc_mean = mean(mlp121_train_acc)*100;
    mlp121_train_acc_std = std(mlp121_train_acc)*100;
    test_acc_mean(3, i) = mlp121_test_acc_mean;
    test_acc_std(3, i) = mlp121_test_acc_std;
%     fprintf("mlp121 results %s: & %.2f/%.2f  & %.2f/%.2f \n",...
%             noise_list(i), mlp121_train_acc_mean, mlp121_train_acc_std, mlp121_test_acc_mean, mlp121_test_acc_std);

    mlp212_test_acc = double(mlp212_test_acc_tsr(end, :));
    clear mlp212_test_acc_tsr;
    mlp212_test_acc_mean = mean(mlp212_test_acc)*100;
    mlp212_test_acc_std = std(mlp212_test_acc)*100;
    mlp212_train_acc = double(mlp212_train_acc_tsr(end, :));
    clear mlp212_train_acc_tsr
    mlp212_train_acc_mean = mean(mlp212_train_acc)*100;
    mlp212_train_acc_std = std(mlp212_train_acc)*100;
    test_acc_mean(4, i) = mlp212_test_acc_mean;
    test_acc_std(4, i) = mlp212_test_acc_std;
%     fprintf("mlp212 results %s: & %.2f/%.2f  & %.2f/%.2f \n",...
%             noise_list(i), mlp212_train_acc_mean, mlp212_train_acc_std, mlp212_test_acc_mean, mlp212_test_acc_std);

    mlp421_test_acc = double(mlp421_test_acc_tsr(end, :));
    clear mlp421_test_acc_tsr;
    mlp421_test_acc_mean = mean(mlp421_test_acc)*100;
    mlp421_test_acc_std = std(mlp421_test_acc)*100;
    mlp421_train_acc = double(mlp421_train_acc_tsr(end, :));
    clear mlp421_train_acc_tsr
    mlp421_train_acc_mean = mean(mlp421_train_acc)*100;
    mlp421_train_acc_std = std(mlp421_train_acc)*100;
    test_acc_mean(5, i) = mlp421_test_acc_mean;
    test_acc_std(5, i) = mlp421_test_acc_std;
%     fprintf("mlp421 results %s: & %.2f/%.2f  & %.2f/%.2f \n",...
%             noise_list(i), mlp421_train_acc_mean, mlp421_train_acc_std, mlp421_test_acc_mean, mlp421_test_acc_std);
   
    mlp12421_test_acc = double(mlp12421_test_acc_tsr(end, :));
    clear mlp12421_test_acc_tsr;
    mlp12421_test_acc_mean = mean(mlp12421_test_acc)*100;
    mlp12421_test_acc_std = std(mlp12421_test_acc)*100;
    mlp12421_train_acc = double(mlp12421_train_acc_tsr(end, :));
    clear mlp12421_train_acc_tsr
    mlp12421_train_acc_mean = mean(mlp12421_train_acc)*100;
    mlp12421_train_acc_std = std(mlp12421_train_acc)*100;
    test_acc_mean(6, i) = mlp12421_test_acc_mean;
    test_acc_std(6, i) = mlp12421_test_acc_std;
%     fprintf("mlp12421 results %s: & %.2f/%.2f  & %.2f/%.2f \n",...
%             noise_list(i), mlp12421_train_acc_mean, mlp12421_train_acc_std, mlp12421_test_acc_mean, mlp12421_test_acc_std);
        
    mlp42124_test_acc = double(mlp42124_test_acc_tsr(end, :));
    clear mlp42124_test_acc_tsr;
    mlp42124_test_acc_mean = mean(mlp42124_test_acc)*100;
    mlp42124_test_acc_std = std(mlp42124_test_acc)*100;
    mlp42124_train_acc = double(mlp42124_train_acc_tsr(end, :));
    clear mlp42124_train_acc_tsr
    mlp42124_train_acc_mean = mean(mlp42124_train_acc)*100;
    mlp42124_train_acc_std = std(mlp42124_train_acc)*100;
    test_acc_mean(7, i) = mlp42124_test_acc_mean;
    test_acc_std(7, i) = mlp42124_test_acc_std;
%     fprintf("mlp42124 results %s: & %.2f/%.2f  & %.2f/%.2f \n",...
%             noise_list(i), mlp42124_train_acc_mean, mlp42124_train_acc_std, mlp42124_test_acc_mean, mlp42124_test_acc_std);
    
    
    cnn11_test_acc = double(cnn11_test_acc_tsr(end, :));
    clear cnn11_test_acc_tsr;
    cnn11_test_acc_mean = mean(cnn11_test_acc)*100;
    cnn11_test_acc_std = std(cnn11_test_acc)*100;
    cnn11_train_acc = double(cnn11_train_acc_tsr(end, :));
    clear cnn11_train_acc_tsr
    cnn11_train_acc_mean = mean(cnn11_train_acc)*100;
    cnn11_train_acc_std = std(cnn11_train_acc)*100;
    test_acc_mean(8, i) = cnn11_test_acc_mean;
    test_acc_std(8, i) = cnn11_test_acc_std;
%     fprintf("cnn11 results %s: & %.2f/%.2f  & %.2f/%.2f \n",...
%             noise_list(i), cnn11_train_acc_mean, cnn11_train_acc_std, cnn11_test_acc_mean, cnn11_test_acc_std);
        
    cnn12_test_acc = double(cnn12_test_acc_tsr(end, :));
    clear cnn12_test_acc_tsr;
    cnn12_test_acc_mean = mean(cnn12_test_acc)*100;
    cnn12_test_acc_std = std(cnn12_test_acc)*100;
    cnn12_train_acc = double(cnn12_train_acc_tsr(end, :));
    clear cnn12_train_acc_tsr
    cnn12_train_acc_mean = mean(cnn12_train_acc)*100;
    cnn12_train_acc_std = std(cnn12_train_acc)*100;
    test_acc_mean(9, i) = cnn12_test_acc_mean;
    test_acc_std(9, i) = cnn12_test_acc_std;
%     fprintf("cnn12 results %s: & %.2f/%.2f  & %.2f/%.2f \n",...
%             noise_list(i), cnn12_train_acc_mean, cnn12_train_acc_std, cnn12_test_acc_mean, cnn12_test_acc_std);
    
    cnn21_test_acc = double(cnn21_test_acc_tsr(end, :));
    clear cnn21_test_acc_tsr;
    cnn21_test_acc_mean = mean(cnn21_test_acc)*100;
    cnn21_test_acc_std = std(cnn21_test_acc)*100;
    cnn21_train_acc = double(cnn21_train_acc_tsr(end, :));
    clear cnn21_train_acc_tsr
    cnn21_train_acc_mean = mean(cnn21_train_acc)*100;
    cnn21_train_acc_std = std(cnn21_train_acc)*100;
    test_acc_mean(10, i) = cnn21_test_acc_mean;
    test_acc_std(10, i) = cnn21_test_acc_std;
%     fprintf("cnn21 results %s: & %.2f/%.2f  & %.2f/%.2f \n",...
%             noise_list(i), cnn21_train_acc_mean, cnn21_train_acc_std, cnn21_test_acc_mean, cnn21_test_acc_std);
    cnn22_test_acc = double(cnn22_test_acc_tsr(end, :));
    clear cnn22_test_acc_tsr;
    cnn22_test_acc_mean = mean(cnn22_test_acc)*100;
    cnn22_test_acc_std = std(cnn22_test_acc)*100;
    cnn22_train_acc = double(cnn22_train_acc_tsr(end, :));
    clear cnn22_train_acc_tsr
    cnn22_train_acc_mean = mean(cnn22_train_acc)*100;
    cnn22_train_acc_std = std(cnn22_train_acc)*100;
    test_acc_mean(11, i) = cnn22_test_acc_mean;
    test_acc_std(11, i) = cnn22_test_acc_std;
%     fprintf("cnn22 results %s: & %.2f/%.2f  & %.2f/%.2f \n",...
%             noise_list(i), cnn22_train_acc_mean, cnn22_train_acc_std, cnn22_test_acc_mean, cnn22_test_acc_std);
%         
%     fprintf("mlp421 results %s: & %.2f/%.2f  & %.2f/%.2f \n",...
%             noise_list(i), mlp421_train_acc_mean, mlp421_train_acc_std, mlp421_test_acc_mean, mlp421_test_acc_std);
    fpn_test_acc = double(fpn_test_acc_tsr(end, :));
    clear fpn_test_acc_tsr;
    fpn_test_acc_mean = mean(fpn_test_acc)*100;
    fpn_test_acc_std = std(fpn_test_acc)*100;
    fpn_train_acc = double(fpn_train_acc_tsr(end, :));
    clear fpn_train_acc_tsr
    fpn_train_acc_mean = mean(fpn_train_acc)*100;
    fpn_train_acc_std = std(fpn_train_acc)*100;
    test_acc_mean(12, i) = fpn_test_acc_mean;
    test_acc_std(12, i) = fpn_test_acc_std;
%     fprintf("fpn results %s: & %.2f/%.2f  & %.2f/%.2f \n",...
%             noise_list(i), fpn_train_acc_mean, fpn_train_acc_std, fpn_test_acc_mean, fpn_test_acc_std);
        
%     fprintf("fpn results (%s): & %.2f/%.2f & %.2f/%.2f & %.2f/%.2f & %.2f/%.2f & %.2f/%.2f & %.2f/%.2f & %.2f/%.2f & %.2f/%.2f & %.2f/%.2f & %.2f/%.2f & %.2f/%.2f & %.2f/%.2f \n",...
%             noise_list(i), svm_test_acc_mean, svm_test_acc_std, mlp21_test_acc_mean, mlp21_test_acc_std, mlp121_test_acc_mean, mlp121_test_acc_std,...
%             mlp212_test_acc_mean, mlp212_test_acc_std, mlp421_test_acc_mean, mlp421_test_acc_std, mlp12421_test_acc_mean, mlp12421_test_acc_std,...
%             mlp42124_test_acc_mean, mlp42124_test_acc_std, cnn11_test_acc_mean, cnn11_test_acc_std, cnn12_test_acc_mean, cnn12_test_acc_std,...
%             cnn21_test_acc_mean, cnn21_test_acc_std, cnn22_test_acc_mean, cnn22_test_acc_std,...
%             fpn_test_acc_mean, fpn_test_acc_std);
        
end
% save es_mean_std.mat test_acc_mean test_acc_std;
for j=1:size(algorithm_list, 2)
    fprintf("%s Acc: & %.2f/%.2f & %.2f/%.2f & %.2f/%.2f & %.2f/%.2f \n",...
            algorithm_list(j), test_acc_mean(j,1), test_acc_std(j,1), test_acc_mean(j,2), test_acc_std(j,2),...
            test_acc_mean(j,3), test_acc_std(j,3), test_acc_mean(j,4), test_acc_std(j,4));
end