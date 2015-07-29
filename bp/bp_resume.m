function [model] = bp_resume(model)
% Resume discriminative finetuning of CDBN.

rng('shuffle');
kernels;
debug = 0;

data_list = read_data_list(model.data_path, model.classnames, ....
    model.volume_size + 2 * model.pad_size, 'train', debug);

param = [];
param.epochs = 100;
param.lr = 0.01;
param.weight_decay = 5*10^-4;
param.momentum = 0.9;
param.batch_size = 32;
param.snapshot_iter = 10;
param.snapshot_name = 'bp_finetune_iter';
param.test_iter = 5;
batch_size = param.batch_size;

fprintf('Resume discriminative funetuning the CDBN\n');
fprintf('lr = %f, wd = %d, momentum = %f\n', param.lr, param.weight_decay, param.momentum);

% prepare data and label
[new_list, label] = balance_data(data_list, batch_size);
n = length(new_list);
batch_num = n / batch_size;
assert(batch_num == floor(batch_num));

% prepare model
numLayer = model.numLayer;
for iter = 1 : param.epochs
    loss_all = 0;
    shuffle_index = randperm(n);
    for b = 1 : batch_num
        batch_index = shuffle_index((b-1)*batch_size + 1 : b * batch_size);
        batch = read_batch(model, new_list(batch_index), false);
        batch_label = label(batch_index,:);
        [model, activation] = bp_forward(model, batch);
        [model, loss] = bp_backward(model, activation, batch_label, param.weight_decay);
        loss_all = loss_all + loss;
        model = bp_update(model, param);
    end
    loss_all = loss_all / batch_num;
    fprintf('iteration: %d, loss: %f\n', iter, loss_all);
    
    if mod(iter, param.snapshot_iter) == 0
        fprintf('snapshoting to %s_%d\n', param.snapshot_name, iter);
        snapshot_name = sprintf('%s_%d', param.snapshot_name, iter);
        save(snapshot_name, 'model');
    end
    
    if mod(iter, param.test_iter) == 0
        test_loss = bp_test(model);
        fprintf('test loss: %f\n', test_loss);
    end
end

for l = 2 : numLayer
    model.layers{l} = rmfield(model.layers{l},'grdw');
    model.layers{l} = rmfield(model.layers{l},'grdc');
    model.layers{l} = rmfield(model.layers{l},'histw');
    model.layers{l} = rmfield(model.layers{l},'histc');
end

save('bp_finetuned_model', 'model');
