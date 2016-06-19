path_c = 'C:\UbuntuShare\ROC\classify\';
file_end = '*csv';
full_c = strcat(path_c, file_end);
files_c = dir(full_c);

C_labels = [];
label_c_length = 0;

% get label length (largest label)
for file_c = files_c'
    c_name = file_c.name;
    c_label = regexprep(c_name, '_', ' ');
    c_label = regexprep(c_label, '.csv', '');
    c_length = length(c_label);
    
    if(c_length > label_c_length)
        label_c_length = c_length;
    end
end

% get matrices of classifier roc
for file_c = files_c'
    % get name + path
    c_name = file_c.name;
    c_label = regexprep(c_name, '_', ' ');
    c_label = regexprep(c_label, '.csv', '');
    c_path = strcat(path_c, c_name);
    c_length = length(c_label);
    
    while label_c_length > c_length
        c_label = sprintf('%s ',c_label);
        c_length = length(c_label);
    end
    
    % read data
    C = dlmread(c_path, '\t');
    C = C.';
    
    C_labels = [C_labels ; c_label];
    
    % plot roc
    [X,Y] = perfcurve(C(1,:), C(2,:), 1.0);
    figure(1); plot(X,Y);
    hold on;
end

xlabel('False positive rate');
ylabel('True positive rate');
title('ROC for Classification by SVM');
legend(C_labels);
hold off;


%get data of detector roc
path_d = 'C:\UbuntuShare\ROC\detect\';
full_d = strcat(path_d, file_end);
files_d = dir(full_d);

D_labels = [];
label_d_length = 0;

% get label length (largest label)
for file_d = files_d'
    d_name = file_d.name;
    d_label = regexprep(d_name, '_', ' ');
    d_label = regexprep(d_label, '.csv', '');
    d_length = length(d_label);
    
    if(d_length > label_d_length)
        label_d_length = d_length;
    end
end

% get matrices of classifier roc
for file_d = files_d'    
    % get name + path
    d_name = file_d.name;
    d_path = strcat(path_d, d_name);
    d_label = regexprep(d_name, '_', ' ');
    d_label = regexprep(d_label, '.csv', '');
    d_length = length(d_label);
    
    while label_d_length > d_length
        d_label = sprintf('%s ',d_label);
        d_length = length(d_label);
    end
    
    % read data
    D = dlmread(d_path, '\t');
    D = D.';
    
    D_labels = [D_labels ; d_label];
    
    % plot roc
    [X,Y] = perfcurve(D(1,:), D(2,:), 1.0);
    figure(2); plot(X,Y);
    hold on;
end
  
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC for Detection by Clustering');
legend(D_labels);
hold off;
        

