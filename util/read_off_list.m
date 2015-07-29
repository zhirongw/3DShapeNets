function data_list = read_off_list(data_path, classes, train_or_test, debug)
% read raw off filenames

% data_path: off data folder
% classes: a cell array of class names
% train_or_test: select training data or testing data('train', 'test')
% debug: true for debug.

if nargin < 4 ; debug = 0; end

if debug
    maxNum = 20;
else
    maxNum = 10000;
end

num_classes = length(classes);
data_list = cell(num_classes,1);
for c = 1 : num_classes
    fprintf('reading the %s category\n', classes{c});
    category_path = [data_path '/' classes{c}  '/off/' train_or_test];
    files = dir(category_path);
    
    cat_ind = 0;
    data_list{c} = [];
    for i = 1 : min(length(files), maxNum)
        if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') || files(i).isdir == 1 || ~strcmp(files(i).name(end-2:end), 'off')
            continue;
        end
        filename = [category_path '/' files(i).name];
        
        cat_ind = cat_ind + 1;
        data_list{c}(cat_ind,1).filename = filename;
        data_list{c}(cat_ind,1).label = c;
        
        if cat_ind == maxNum; break; end
    end
end

