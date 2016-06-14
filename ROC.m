path_c = 'C:\UbuntuShare\ROC\classify\';
file_end = '*csv';
full_c = strcat(path_c, file_end);
files_c = dir(full_c);

C_targets = [];
C_outputs = [];
C_labels = [];

% get matrices of classifier roc
for file_c = files_c'
    % get name + path
    c_name = file_c.name;
    c_label = regexprep(c_name, '_', ' ');
    c_label = regexprep(c_label, '.csv', '');
    c_path = strcat(path_c, c_name);
    
    % read data
    C = dlmread(c_path, '\t');
    C = C.';
    
    C_labels = [C_labels ; c_label];
    C_targets = [C_targets ; C(1,:)];
    C_outputs = [C_outputs ; C(2,:)];
end

% get data of detector roc
path_d = 'C:\UbuntuShare\ROC\detect\';
full_d = strcat(path_d, file_end);
files_d = dir(full_d);


D_targets = [];
D_outputs = [];
D_labels = [];

% get matrices of classifier roc
for file_d = files_d'    
    % get name + path
    d_name = file_d.name;
    d_path = strcat(path_d, d_name);
    
    % read data
    D = dlmread(d_path, '\t');
    D = D.';
    
    D_labels = [D_labels ; d_name];
    D_targets = [D_targets ; D(1,:)];
    D_outputs = [D_outputs ; D(2,:)];
end
  
plotroc(C_targets, C_outputs, 'Classifier', D_targets, D_outputs, 'Detector');
legend('','Hard Negative Mining', '', 'Without Hard Negative Mining');



%M = dlmread('E:\Benutzer\Daniel\Desktop\test.csv', '\t')
%M = dlmread('E:\Benutzer\Daniel\Desktop\UbuntuShare\_result1.csv', '\t')
%M2 = dlmread('E:\Benutzer\Daniel\Desktop\UbuntuShare\_result2.csv', '\t')

%M = M.'
%targets = M(1,:)
%outputs = M(2,:)

%M2 = M2.'
%targets2 = M2(1,:)
%outputs2 = M2(2,:)

%plotroc(targets, outputs, 'Sliding Window', targets2, outputs2, 'Merged Contour')





%run('C:\Users\Kevin\Documents\MATLAB\vlfeat-master\toolbox\vl_setup');
%[recall, precision] = vl_pr(D_targets, D_outputs);
%vl_roc(D_targets, D_outputs);

%vl_roc(D_targets, D_outputs);
%[TPR,TNR,INFO] = vl_roc(D_targets, D_outputs);

%N = size(files_c, 1)
%hL = zeros(N,1);
%for k=1:N
%        hL(k) = plot(1.0 - TNR, TPR);
%end
%hLeg = legend(hL, 'Höbi stinkt');
        

