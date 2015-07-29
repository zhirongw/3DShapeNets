function [new_list, label] = balance_data(data_list, batch_size)
% balance categories with dramatically different number of data by
% duplicating the data of small categories so that each category has
% roughly the same number of data.
% data_list: A cell-array of classes. Each cell contains a array of
% filenames of traning data. Returned by read_data_list.m
% new_list: A array of filenames mixed with different classes of equal
% number of data.
% label: corresponding labels for new_list.

numClass = length(data_list);
numPerClass = zeros(numClass,1);
for c = 1 : numClass
    numPerClass(c) = length(data_list{c});
end

maxNumPerClass = max(numPerClass);
numBatches = floor(maxNumPerClass * numClass / batch_size);
while maxNumPerClass * numClass ~= numBatches * batch_size
    maxNumPerClass = floor(numBatches * batch_size / numClass);
    numBatches = floor(maxNumPerClass * numClass / batch_size);
end

numTotal = maxNumPerClass * numClass;
new_list = repmat(struct('filename', '', 'label', 0), numTotal, 1);
label = zeros(numTotal,1);
for c = 1 : numClass
    label(maxNumPerClass*(c-1)+1:maxNumPerClass*c) = c;
    if numPerClass(c) >= maxNumPerClass
        new_list(maxNumPerClass*(c-1)+1 : maxNumPerClass*c) = data_list{c}(1 : maxNumPerClass);
    else
        mult = floor(maxNumPerClass / numPerClass(c));
        new_list(maxNumPerClass*(c-1)+1 : maxNumPerClass*(c-1)+numPerClass(c)*mult) = repmat(data_list{c}, mult, 1);
        residue = maxNumPerClass - numPerClass(c) * mult;
        shuffle_index = randperm(numPerClass(c));
        new_list(maxNumPerClass*(c-1)+numPerClass(c)*mult+1: maxNumPerClass*c) = data_list{c}(shuffle_index(1:residue));
    end
end
label = single(full(sparse(1:numTotal,label,1)));
