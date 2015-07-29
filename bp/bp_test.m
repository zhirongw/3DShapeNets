function loss_all = bp_test(model)

data_list = read_data_list(model.data_path, model.classnames, ...
    model.volume_size + 2 * model.pad_size, 'test');

% create testing data list and labels
label = [];
new_list = repmat(struct('filename', '', 'label', 0), 1, 1);
curr = 1;
for i = 1 : model.classes
    cnt = length(data_list{i});
    label(curr:curr+cnt-1,1) = i;
    new_list(curr:curr+cnt-1,1) = data_list{i};
    curr = curr + cnt;
end
n = length(label);

% prepare data and label
test_index = [1:n];
batch_size = 32;
batch_num = ceil(n / batch_size);

accuracy = 0;
loss_all = 0;
for b = 1 : batch_num
    idx_end = min(b*batch_size, n);
    batch_index = test_index((b-1)*batch_size + 1 : idx_end);
    batch = read_batch(model, new_list(batch_index), false);
    batch_label = label(batch_index,:);
    [model, activation] = bp_forward(model, batch);
    
    [~, predict] = max(activation{end}, [], 2);
    accuracy = accuracy + sum(predict == batch_label);
    loss = - sum( log(activation{end}(full(sparse(1:length(batch_index), batch_label, 1)) == 1) ) );
    loss_all = loss_all + loss;
end

accuracy = accuracy / n;
loss_all = loss_all / n;
fprintf('test accuracy: %f\n', accuracy);
