function retrieval_eval(model)

% first prepare the feature
allClasses = model.classnames;
numClasses = numel(allClasses);

% the path of precomputed feature
feat_path = model.data_path;
data_size = model.volume_size + 2 * model.pad_size;
numInstPerClass = 240;

test_data = [];
test_label = [];
for c = 1 : numClasses
    the_class = allClasses{c};
    class_path = [feat_path '/' the_class '/' num2str(data_size)];
    load([class_path '/test_feature']);
    c_num = size(features,1);
    assert(c_num == numInstPerClass);
    test_data(end+1:end+c_num,:) = features;
    test_label(end+1:end+c_num, 1) = c;
end

% step 2: calculate pairwise L2 distance
distance_matrix = zeros(size(test_data,1));
for r = 1 : size(test_data,1)
    for c = r+1 : size(test_data, 1)
        distance_matrix(r,c) = norm(test_data(r,:) - test_data(c,:));
        distance_matrix(c,r) = distance_matrix(r,c);
    end
end

% step 3: sort the retrievals
label_matrix = zeros(size(test_data,1));
for i = 1 : size(distance_matrix,1)
    [~, ind] = sort(distance_matrix(i,:), 'ascend');
    label_matrix(i,:) = test_label(ind);
end

% evaluation numbers
numInst = size(label_matrix, 1);
% raw recall and precision
recall = zeros(numInst, numInst);
precision = zeros(numInst, numInst);
% precision at each positive retrieval points
trec_precision = zeros(numInst, numInstPerClass);
% interpolated recall and precision at all points
mrecall = zeros(numInst, numInst+2);
mprecision = zeros(numInst, numInst+2);
% average precision
ap = zeros(numInst, 1);
for r = 1 : size(recall,1)
    tp = 0;
    for c = 1 : size(recall,2)
        if label_matrix(r,c) == test_label(r)
            tp = tp + 1;
            trec_precision(r,tp) = tp / c;
        end
        recall(r,c) = tp / numInstPerClass;
        precision(r,c) = tp / c;
    end
    % interpolate it
    mrec=[0 , recall(r,:) , 1];
    mpre=[0 , precision(r,:) , 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap(r)=sum((mrec(i)-mrec(i-1)).*mpre(i));
    mrecall(r,:) = mrec;
    mprecision(r,:) = mpre;
end

% two ap measures
mean_ap_auc = mean(ap)
mean_ap_trec = mean(mean(trec_precision,2))

% one pr plot
pr_curve = zeros(numInst, 10);
for r = 1 : size(label_matrix,1)
    this_mprec = mprecision(r,:);
    for c = 1 : 10
        pr_curve(r,c) = max(this_mprec(mrecall(r,:) > (c-1)*0.1));
    end
end

res = [];
res.recall = recall; res.precision = precision;
res.trec_precision = trec_precision;
res.pr_curve = pr_curve;
